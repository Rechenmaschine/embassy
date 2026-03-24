//! Async interface for dispatching `FnOnce(&mut T) -> R` jobs to a dedicated
//! runner task with exclusive access to `T`.
//!
//! Callers submit an `FnOnce(&mut T) -> R` via [`ContextService::call`].
//! A dedicated runner, started with [`ContextService::run`], executes
//! closures one at a time with exclusive `&mut T` access and sends
//! results back. Closures and return values are stored inline in a
//! fixed-size slot of `S` bytes, checked at compile time.

use core::cell::{Cell, UnsafeCell};
use core::future::Future;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll};

use crate::blocking_mutex::Mutex;
use crate::blocking_mutex::raw::RawMutex;
use crate::signal::Signal;
use crate::waitqueue::WakerRegistration;

type RunFn<T, const S: usize> = unsafe fn(&Storage<S>, &mut T);

// For various reasons, one might want to defer code from an async context
// to a blocking context.
//
// We would like to enable async callers to run FnOnce(&mut T) -> R on a shared
// T, where T is owned by a single runner task. The interface should remain simple,
// with a call() function to submit a closure and a run() function to drive execution.
//
// ## Design requirements
// Ideally, each call should be able to have different F and R types. Since the
// runner task will be unaware of the exact F and R, we will need to erase the type
// of the closure and its return value.
//
// Like the existing primitives in embassy-sync, both call() and run()
// should be cancel-safe: dropping either future at any await point must
// leave the service in a consistent state and ready for further use.
//
// Particularly, when a call future is dropped, either:
//   - the closure was not yet submitted and is simply dropped, or
//   - the closure was already submitted and will be executed by the
//     runner to completion (the result is discarded).
// In both cases, the service remains usable and applies backpressure
// correctly: new callers are simply blocked until the slot is free.
//
// Since run() can also be cancelled, it would be reasonable for it to also
// be restartable, that is, a new run() call must be able to pick up where the
// previous one left off, recovering any in-flight state before accepting new work.
//
// ## Implementation
// Fundamentally, we need some shared memory between the caller and the runner for the
// closure (and its captured environment) and the return value.
//
// Stack-pinned memory on the caller side is not an option since after
// submission, the runner may still be reading F or writing R when the
// caller is dropped. We cannot block in Drop to wait for it to finish,
// and also cannot interrupt the runner mid-execution either.
//
// Instead of living on the stack, our closure and result will live in a shared
// fixed-size byte buffer ("slot"), owned by the ContextService with access
// coordinated by a handshake protocol.
//
// Type erasure:
//   The slot (Storage<S>) is an S-byte buffer. The caller writes its
//   closure F into the slot via a pointer cast, then sends a function
//   pointer run_job::<T, R, F, S> through the job signal. The runner
//   calls this function pointer, which knows the concrete types: it
//   takes F out of the slot, calls it, and writes R back. This lets
//   the runner loop stay non-generic over F and R.
//
// Coordination uses a SlotState (behind a blocking mutex) and three signals:
//
//   - state (Mutex<Cell<SlotState>>): a boolean `free` flag and a
//     WakerRegistration. When a caller tries to acquire the slot and it
//     is not free, the caller registers its waker here and returns
//     Pending. When the runner finishes a job and marks the slot free,
//     it wakes the registered waker. This is the same pattern that
//     Channel uses for backpressure (senders_waker / receiver_waker).
//   - job (Signal<RunFn>): caller -> runner. Carries the type-erased
//     function pointer and wakes the runner to start executing.
//   - done (Signal<()>): runner -> caller. Tells the caller that R is
//     ready in the slot.
//   - ack (Signal<()>): caller -> runner. Tells the runner the caller is
//     done reading R and the slot can be cleaned up.
//
// The full handshake:
//
//   caller                           runner
//     |                                |
//     |---- acquire slot ------------->|
//     |       store F into slot        |
//     |---- signal job --------------->|
//     |                          take F, execute it, store R
//     |<--- signal done ---------------|
//     |       take R from slot         |
//     |---- signal ack --------------->|
//     |                          drop slot contents, mark slot free
//     |                                |
//
// Slot ownership follows during the handshake:
//   - caller owns the slot between acquiring it and signalling job
//   - runner owns the slot between receiving job and signalling done
//   - caller owns the slot between receiving done and signalling ack
//   - runner owns the slot between receiving ack and marking the slot free
//
// To support cancellation and restartability, the handshake is extended
// with a few atomic flags:
//
//   caller                          runner
//     |                               |
//     |                         [running = true, drop guard armed]
//     |                         [if !initialized: mark slot free]
//     |                               |
//     |--- acquire slot ------------->|
//     |    store F in slot            |
//     |--- signal job --------------->|
//     |                         [needs_recovery = true]
//     |                         take F, execute it, store R
//     |<-- signal done ---------------|
//     |    take R from slot           |
//     |--- signal ack --------------->|
//     |                         drop slot contents
//     |                         [needs_recovery = false]
//     |                         mark slot free
//     |                               |
//
//   The runner can be dropped at any await point (job.wait, ack.wait).
//   If dropped while needs_recovery is true, the slot may still contain
//   data and the caller may still be active. The next run() checks
//   needs_recovery, waits for the caller's ack, and cleans up before
//   entering the main loop.
//
//   running (AtomicBool): prevents concurrent runners. Cleared by a
//     drop guard so a new run() can start after the old one is dropped.
//
// The key invariant is that every job signal is eventually followed by an
// ack signal, no matter what gets dropped. CallFuture::drop sends ack if
// the closure was already submitted. This guarantees the runner (or the
// next runner, after recovery) can always make progress.
//
// Assumptions about Signal semantics (embassy_sync::signal::Signal):
//   - signal(v) is sticky: if signaled before wait(), the value is not lost.
//   - wait() consumes exactly one value: returns Ready once, then resets to None.
//   - signal() overwrites any unconsumed value (single-slot, last-writer-wins).
//   - signal()/wait() provide at least release/acquire synchronization:
//     writes before signal() are visible after wait() returns.
//   - wait() is cancel-safe: dropping the future does not consume the signal.
//   - reset() clears any unconsumed value.
//
// Closures must not panic. Under panic=unwind, a panic in f(state) leaves
// needs_recovery set and done unsignaled. The caller blocks until dropped,
// at which point its Drop sends ack and the next runner can recover. This
// is acceptable for embedded (panic=abort) but not robust under unwinding.

struct SlotState {
    free: bool,
    waker: WakerRegistration,
}

impl SlotState {
    const EMPTY: Self = Self {
        free: false,
        waker: WakerRegistration::new(),
    };
}

/// Type-erased storage for closures and return values.
///
/// Invariants:
/// - `drop_fn = Some(drop_glue::<T>)` iff the slot contains a live `T`
/// - `store<T>()` writes a `T` and arms `drop_fn`
/// - `take<T>()` reads a `T` out and clears `drop_fn`
/// - `drop_contents()` drops in place if occupied; no-op if empty
#[repr(C, align(8))]
struct Storage<const S: usize> {
    buf: UnsafeCell<MaybeUninit<[u8; S]>>,
    drop_fn: UnsafeCell<Option<unsafe fn(&Self)>>,
}

impl<const S: usize> Storage<S> {
    const fn new() -> Self {
        Self {
            buf: UnsafeCell::new(MaybeUninit::uninit()),
            drop_fn: UnsafeCell::new(None),
        }
    }

    /// # Safety
    /// Slot must be empty. `size_of::<T>() <= S` and `align_of::<T>() <= align_of::<Self>()`.
    unsafe fn store<T>(&self, val: T) {
        // SAFETY: caller guarantees the slot is empty and T fits.
        unsafe {
            (*self.buf.get()).as_mut_ptr().cast::<T>().write(val);
            *self.drop_fn.get() = Some(Self::drop_glue::<T>);
        }
    }

    /// # Safety
    /// Slot must contain a live `T`.
    unsafe fn take<T>(&self) -> T {
        // SAFETY: caller guarantees a live T is in the slot.
        unsafe {
            let val = (*self.buf.get()).as_ptr().cast::<T>().read();
            *self.drop_fn.get() = None;
            val
        }
    }

    /// # Safety
    /// Caller must have exclusive access.
    ///
    /// # Panics
    /// If the stored value's destructor panics, `drop_fn` is already
    /// cleared so double-drop will not occur. The buffer bytes remain
    /// and are overwritten by the next `store()`.
    unsafe fn drop_contents(&self) {
        // SAFETY: caller guarantees exclusive access.
        unsafe {
            if let Some(f) = (*self.drop_fn.get()).take() {
                f(self);
            }
        }
    }

    /// # Safety
    /// `slot` must currently contain a live `T`.
    unsafe fn drop_glue<T>(slot: &Self) {
        // SAFETY: caller guarantees the slot contains a live T.
        unsafe {
            core::ptr::drop_in_place((*slot.buf.get()).as_mut_ptr().cast::<T>());
        }
    }
}

impl<const S: usize> Drop for Storage<S> {
    fn drop(&mut self) {
        // SAFETY: &mut self guarantees exclusive access.
        unsafe { self.drop_contents() };
    }
}

/// # Safety
/// - `slot` must currently contain a live `F`.
/// - `R` must fit in the slot.
///
/// After return, slot contains a live `R`.
///
/// # Panics
/// If `f(state)` panics, `F` has already been taken from the slot and `R` is
/// never stored. The slot is left empty (`drop_fn` is `None`). Under unwinding,
/// the caller waiting on `done` will block until dropped.
unsafe fn run_job<T, R, F: FnOnce(&mut T) -> R, const S: usize>(slot: &Storage<S>, state: &mut T) {
    // SAFETY: caller guarantees slot contains a live F and R fits.
    unsafe {
        let f: F = slot.take();
        let res = f(state);
        slot.store(res);
    }
}

struct JobSlot<M: RawMutex, T, const S: usize> {
    storage: Storage<S>,
    state: Mutex<M, Cell<SlotState>>,
    job: Signal<M, RunFn<T, S>>,
    done: Signal<M, ()>,
    ack: Signal<M, ()>,
}

impl<M: RawMutex, T, const S: usize> JobSlot<M, T, S> {
    const fn new() -> Self {
        Self {
            storage: Storage::new(),
            state: Mutex::new(Cell::new(SlotState::EMPTY)),
            job: Signal::new(),
            done: Signal::new(),
            ack: Signal::new(),
        }
    }

    fn debug_assert_held(&self) {
        self.state.lock(|cell| {
            let s = cell.replace(SlotState::EMPTY);
            debug_assert!(!s.free, "slot accessed without being held");
            cell.set(s);
        });
    }

    fn poll_acquire(&self, cx: &mut Context<'_>) -> Poll<()> {
        self.state.lock(|cell| {
            let mut s = cell.replace(SlotState::EMPTY);
            if s.free {
                s.free = false;
                cell.set(s);
                Poll::Ready(())
            } else {
                s.waker.register(cx.waker());
                cell.set(s);
                Poll::Pending
            }
        })
    }

    fn try_acquire(&self) -> bool {
        self.state.lock(|cell| {
            let mut s = cell.replace(SlotState::EMPTY);
            if s.free {
                s.free = false;
                cell.set(s);
                true
            } else {
                cell.set(s);
                false
            }
        })
    }

    /// # Safety
    /// Caller must have acquired the slot. F and R must fit.
    unsafe fn submit<R, F: FnOnce(&mut T) -> R>(&self, f: F) {
        // SAFETY: caller guarantees slot is acquired and F/R fit.
        unsafe { self.storage.store(f) };
        self.job.signal(run_job::<T, R, F, S>);
    }

    /// # Safety
    /// Caller must have acquired the slot. F must fit.
    unsafe fn submit_immediate<F: FnOnce(&mut T)>(&self, f: F) {
        // SAFETY: caller guarantees slot is acquired and F fits.
        unsafe { self.storage.store(f) };
        self.ack.signal(());
        self.job.signal(run_job::<T, (), F, S>);
    }

    fn mark_free(&self) {
        self.state.lock(|cell| {
            let mut s = cell.replace(SlotState::EMPTY);
            s.free = true;
            s.waker.wake();
            cell.set(s);
        });
    }

    fn poll_result<R>(&self, cx: &mut Context<'_>) -> Poll<R> {
        let mut fut = self.done.wait();
        match Pin::new(&mut fut).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(()) => {
                self.debug_assert_held();
                // SAFETY: done was signaled, so the runner wrote R into the slot.
                let result = unsafe { self.storage.take::<R>() };
                self.ack.signal(());
                Poll::Ready(result)
            }
        }
    }

    /// Wait for the caller's ack, then clean up the slot and mark it free.
    ///
    /// If the stored value's destructor panics under unwinding, the slot
    /// is still reset and freed. The destructor's side effects are lost
    /// but the buffer is reused by the next job; no double-drop occurs.
    async fn wait_ack_and_finish(&self, needs_recovery: &AtomicBool) {
        // Drop guard ensures done.reset(), mark_free(), and needs_recovery
        // cleanup run even if drop_contents() panics. Same pattern as
        // multi_waker's set_len(0)-before-wake: if the destructor panics,
        // its side effects are lost but the service remains usable.
        struct FinishGuard<'a, M: RawMutex, T, const S: usize> {
            slot: &'a JobSlot<M, T, S>,
            needs_recovery: &'a AtomicBool,
        }
        impl<M: RawMutex, T, const S: usize> Drop for FinishGuard<'_, M, T, S> {
            fn drop(&mut self) {
                self.needs_recovery.store(false, Ordering::Release);
                self.slot.done.reset();
                self.slot.mark_free();
            }
        }

        self.ack.wait().await;
        self.debug_assert_held();

        let _guard = FinishGuard {
            slot: self,
            needs_recovery,
        };
        // SAFETY: ack received, so the caller is done with the slot.
        // drop_contents clears drop_fn before calling the destructor,
        // so a panic here won't cause double-drop.
        unsafe { self.storage.drop_contents() };
    }
}

enum Phase {
    Acquiring,
    Submitted,
    Done,
}

/// Dispatch closures for execution on a dedicated runner task.
///
/// Callers submit an `FnOnce(&mut T) -> R` via [`call`](ContextService::call).
/// The runner, started with [`run`](ContextService::run), executes closures one at a
/// time with exclusive `&mut T` access and sends results back. Each call
/// can return a different type.
///
/// Closures and return values are stored inline in a fixed-size slot of `S` bytes.
/// Both the closure and the return type must fit within `S` bytes and require no
/// more alignment than the slot. This is checked at compile time.
///
/// ## Example
///
/// ```rust,ignore
/// static FS: ContextService<CriticalSectionRawMutex, Filesystem, 64> =
///     ContextService::new();
///
/// // runner task
/// FS.run(&mut filesystem).await;
///
/// // caller task
/// let size = FS.call(|fs| fs.read_blocking(path).len()).await;
/// ```
pub struct ContextService<M: RawMutex, T, const S: usize> {
    slot: JobSlot<M, T, S>,
    running: AtomicBool,
    initialized: AtomicBool,
    needs_recovery: AtomicBool,
}

impl<M: RawMutex, T, const S: usize> ContextService<M, T, S> {
    /// Create a new `ContextService`.
    pub const fn new() -> Self {
        Self {
            slot: JobSlot::new(),
            running: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
            needs_recovery: AtomicBool::new(false),
        }
    }

    /// Submit a closure for execution on the runner and await the result.
    ///
    /// Fails at compile time if `F` or `R` exceeds the slot capacity `S`.
    ///
    /// The returned future is cancel-safe: dropping it at any point is
    /// sound and leaves the service in a usable state. If dropped before
    /// the closure has been submitted to the runner, no work is performed.
    /// If dropped after submission, the runner will execute the closure to
    /// completion and the return value is discarded.
    pub fn call<R, F>(&self, f: F) -> CallFuture<'_, M, T, R, F, S>
    where
        F: FnOnce(&mut T) -> R + Send + 'static,
        R: Send + 'static,
    {
        const { assert_slot_fits::<F, R, S>() };

        CallFuture {
            svc: self,
            f: Some(f),
            phase: Phase::Acquiring,
            _marker: PhantomData,
        }
    }

    /// Try to submit a fire-and-forget closure without blocking.
    ///
    /// Returns `true` if the closure was submitted, `false` if the slot is busy.
    /// The closure will be executed by the runner; there is no way to retrieve
    /// a return value.
    pub fn try_call_immediate<F>(&self, f: F) -> bool
    where
        F: FnOnce(&mut T) + Send + 'static,
    {
        const { assert_slot_fits::<F, (), S>() };

        if !self.slot.try_acquire() {
            return false;
        }

        // SAFETY: we just acquired the slot, F fits (compile-time check above).
        unsafe { self.slot.submit_immediate(f) };
        true
    }

    /// Run the service loop, executing closures submitted via [`call`](Self::call)
    /// with exclusive `&mut T` access.
    ///
    /// # Panics
    ///
    /// Panics if called while another runner is still active.
    /// Sequential calls after a previous runner was dropped are fine.
    ///
    /// # Cancellation
    ///
    /// This future is cancel-safe. A subsequent call to `run()` will recover any
    /// in-flight state and resume processing. Callers that were blocked will
    /// transparently continue once the new runner starts.
    pub async fn run(&self, state: &mut T) -> ! {
        struct RunGuard<'a> {
            running: &'a AtomicBool,
        }
        impl Drop for RunGuard<'_> {
            fn drop(&mut self) {
                self.running.store(false, Ordering::Release);
            }
        }

        if self.running.swap(true, Ordering::Acquire) {
            panic!("ContextService::run() must not be called concurrently")
        }
        let _guard = RunGuard {
            running: &self.running,
        };

        // If the previous runner was cancelled mid-job, the caller may still
        // be interacting with the slot. Wait for it to finish (the caller
        // always acks, either explicitly or via its Drop), then clean up.
        if self.needs_recovery.load(Ordering::Acquire) {
            self.slot.wait_ack_and_finish(&self.needs_recovery).await;
        }

        // Mark the slot as free exactly once across all run() calls.
        // On restarts without recovery, the slot is either already free
        // (previous finish freed it) or contains a pending job that
        // the runner will pick up via wait_job below.
        if !self.initialized.swap(true, Ordering::Relaxed) {
            self.slot.mark_free();
        }

        loop {
            // Wait for a caller to submit a closure.
            // This is a clean cancellation point: no job is in flight.
            let run_fn = self.slot.job.wait().await;

            // Mark in-flight so a subsequent run() knows to recover.
            self.needs_recovery.store(true, Ordering::Release);

            // SAFETY: slot contains a live F, run_fn matches its types.
            // No other task can access the slot: the caller is waiting on done.
            // Note: if the closure panics under unwinding, done is never signaled.
            // The caller blocks until dropped, then ack allows recovery.
            unsafe { run_fn(&self.slot.storage, state) };
            self.slot.done.signal(());

            // Wait for the caller to read R and signal ack (or for
            // CallFuture::drop to signal ack on cancellation), then
            // clean up the slot and mark it free for the next caller.
            // If cancelled here, needs_recovery is true and the next
            // run() will wait for ack before touching the slot.
            self.slot.wait_ack_and_finish(&self.needs_recovery).await;
        }
    }
}

// SAFETY: access to Storage is serialized by the call/run handshake protocol.
// The remaining fields (signals, mutex, atomics) are Sync on their own.
unsafe impl<M: RawMutex, T, const S: usize> Sync for ContextService<M, T, S>
where
    Mutex<M, Cell<SlotState>>: Sync,
    Signal<M, RunFn<T, S>>: Sync,
    Signal<M, ()>: Sync,
{
}

/// Future returned by [`ContextService::call`].
///
/// This future is cancel-safe. See [`ContextService::call`] for details.
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct CallFuture<'a, M: RawMutex, T, R, F, const S: usize> {
    svc: &'a ContextService<M, T, S>,
    f: Option<F>,
    phase: Phase,
    _marker: PhantomData<R>,
}

impl<M: RawMutex, T, R, F, const S: usize> Unpin for CallFuture<'_, M, T, R, F, S> {}

impl<M, T, R, F, const S: usize> Future for CallFuture<'_, M, T, R, F, S>
where
    M: RawMutex,
    F: FnOnce(&mut T) -> R + Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        loop {
            match self.phase {
                Phase::Acquiring => match self.svc.slot.poll_acquire(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(()) => {
                        let f = self.f.take().unwrap();
                        // SAFETY: we just acquired the slot, F and R fit (compile-time check in call()).
                        unsafe { self.svc.slot.submit::<R, F>(f) };
                        self.phase = Phase::Submitted;
                    }
                },
                Phase::Submitted => return match self.svc.slot.poll_result::<R>(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(result) => {
                        self.phase = Phase::Done;
                        Poll::Ready(result)
                    }
                },
                Phase::Done => panic!("CallFuture polled after completion"),
            }
        }
    }
}

impl<M: RawMutex, T, R, F, const S: usize> Drop for CallFuture<'_, M, T, R, F, S> {
    fn drop(&mut self) {
        if matches!(self.phase, Phase::Submitted) {
            // Future dropped after the job was submitted. The runner will still
            // finish executing the closure. We cannot touch the slot (the runner
            // may still be using it) and we cannot block. Signal ack so the
            // runner can clean up and accept new work.
            self.svc.slot.ack.signal(());
        }
    }
}

const fn assert_slot_fits<F, R, const S: usize>() {
    assert!(
        mem::size_of::<F>() <= S,
        "closure must fit in slot, increase S"
    );
    assert!(
        mem::size_of::<R>() <= S,
        "return type must fit in slot, increase S"
    );
    assert!(
        mem::align_of::<F>() <= mem::align_of::<Storage<S>>(),
        "closure alignment must not exceed 8 bytes"
    );
    assert!(
        mem::align_of::<R>() <= mem::align_of::<Storage<S>>(),
        "return type alignment must not exceed 8 bytes"
    );
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::String;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex};
    use futures_executor::block_on;
    use futures_util::pin_mut;

    #[futures_test::test]
    async fn basic() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            svc.call(|s| {
                *s += 10;
                *s
            })
            .await
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        match futures_util::future::select(caller, runner).await {
            futures_util::future::Either::Left((r, _)) => assert_eq!(r, 10),
            _ => panic!(),
        }
    }

    #[futures_test::test]
    async fn sequential() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            for i in 1..=5 {
                assert_eq!(
                    svc.call(|s| {
                        *s += 1;
                        *s
                    })
                    .await,
                    i
                );
            }
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn different_return_types() {
        let svc: ContextService<NoopRawMutex, Vec<String>, 256> = ContextService::new();
        let mut state = Vec::new();
        let caller = async {
            svc.call(|s| s.push("hello".into())).await;
            assert_eq!(svc.call(|s| s.len()).await, 1);
            assert_eq!(svc.call(|s| s.join(", ")).await, "hello");
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn cancel_before_submit() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        // Drop the future without awaiting. No runner is active, so the
        // closure was never submitted; dropping should be harmless.
        let _fut = svc.call(|s| {
            *s += 1;
            *s
        });
    }

    #[futures_test::test]
    async fn cancel_before_acquire_then_next_call() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            {
                let fut = svc.call(|s: &mut i32| {
                    *s += 100;
                    *s
                });
                pin_mut!(fut);
                // Poll once; slot is not free yet (no runner), so Pending.
                assert!(futures_util::poll!(&mut fut).is_pending());
                // Drop the future. The closure was never submitted.
            }
            svc.call(|s| {
                *s += 1;
                *s
            })
            .await
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        match futures_util::future::select(caller, runner).await {
            futures_util::future::Either::Left((r, _)) => assert_eq!(r, 1),
            _ => panic!(),
        }
    }

    #[futures_test::test]
    async fn closure_dropped_on_cancel_before_acquire() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let dc = drop_count.clone();

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        {
            let tracked = Tracked(dc);
            let fut = svc.call(move |_| {
                let _ = &tracked;
            });
            pin_mut!(fut);
            assert!(futures_util::poll!(&mut fut).is_pending());
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
    }

    #[futures_test::test]
    async fn const_init() {
        static SVC: ContextService<CriticalSectionRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            SVC.call(|s| {
                *s += 5;
                *s
            })
            .await
        };
        let runner = SVC.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        match futures_util::future::select(caller, runner).await {
            futures_util::future::Either::Left((r, _)) => assert_eq!(r, 5),
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "must not be called concurrently")]
    fn concurrent_run_panics() {
        block_on(async {
            let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
            let mut s1 = 0;
            let mut s2 = 0;
            let a = svc.run(&mut s1);
            let b = svc.run(&mut s2);
            pin_mut!(a);
            pin_mut!(b);
            let _ = futures_util::poll!(&mut a);
            let _ = futures_util::poll!(&mut b);
        });
    }

    #[futures_test::test]
    async fn restart_idle() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;

        {
            let caller = async {
                svc.call(|s| {
                    *s += 10;
                    *s
                })
                .await
            };
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 10),
                _ => panic!(),
            }
        }

        {
            let caller = async {
                svc.call(|s| {
                    *s += 5;
                    *s
                })
                .await
            };
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 15),
                _ => panic!(),
            }
        }
    }

    #[futures_test::test]
    async fn restart_after_cancel_mid_job() {
        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        let mut state = ();

        let drop_count = Arc::new(AtomicUsize::new(0));

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let _ = futures_util::poll!(&mut runner);

            {
                let dc = drop_count.clone();
                let fut = svc.call(move |_| Tracked(dc));
                pin_mut!(fut);
                let _ = futures_util::poll!(&mut fut);
                let _ = futures_util::poll!(&mut runner);
                // Drop caller: its Drop signals ack.
            }
            // Runner is dropped here while needs_recovery is true.
        }

        // Tracked return value should not have been dropped yet
        // (it's in the slot, waiting for recovery).
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);

        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let _ = futures_util::poll!(&mut runner);

            assert_eq!(drop_count.load(Ordering::Relaxed), 1);

            let caller = async { svc.call(|_| 42u32).await };
            pin_mut!(caller);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 42),
                _ => panic!(),
            }
        }
    }

    #[futures_test::test]
    async fn restart_many_cycles() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;

        for _ in 0..10 {
            let caller = async {
                svc.call(|s| {
                    *s += 1;
                    *s
                })
                .await
            };
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            futures_util::future::select(caller, runner).await;
        }

        assert_eq!(state, 10);
    }

    #[futures_test::test]
    async fn large_capture() {
        let svc: ContextService<NoopRawMutex, Vec<u8>, 256> = ContextService::new();
        let mut state = Vec::new();
        let data = [42u8; 128];
        let caller = async {
            svc.call(move |s| {
                s.extend_from_slice(&data);
                s.len()
            })
            .await
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        match futures_util::future::select(caller, runner).await {
            futures_util::future::Either::Left((len, _)) => assert_eq!(len, 128),
            _ => panic!(),
        }
    }

    #[futures_test::test]
    async fn zero_sized_return() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            svc.call(|s| {
                *s += 1;
            })
            .await
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn drop_glue_on_service_drop() {
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct Payload(Arc<AtomicUsize>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
            let mut state = ();
            let dc = drop_count.clone();
            let caller = async {
                let fut = svc.call(move |_| Payload(dc));
                pin_mut!(fut);
                assert!(futures_util::poll!(&mut fut).is_pending());
            };
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let _ = futures_util::poll!(&mut runner);
            pin_mut!(caller);
            let _ = futures_util::poll!(&mut caller);
            let _ = futures_util::poll!(&mut runner);
            drop(caller);
            let _ = futures_util::poll!(&mut runner);
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
    }

    #[futures_test::test]
    async fn heterogeneous_types() {
        let svc: ContextService<NoopRawMutex, Vec<u8>, 256> = ContextService::new();
        let mut state = vec![1u8, 2, 3];
        let caller = async {
            let len: usize = svc.call(|s| s.len()).await;
            assert_eq!(len, 3);
            let s: String = svc.call(|s| String::from_utf8(s.clone()).unwrap_or_default()).await;
            assert_eq!(s, "\x01\x02\x03");
            svc.call(|s| s.clear()).await;
            let empty: bool = svc.call(|s| s.is_empty()).await;
            assert!(empty);
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn cancel_after_submit_no_leak() {
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct Heavy(Arc<AtomicUsize>, #[allow(dead_code)] [u8; 64]);
        impl Drop for Heavy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let svc: ContextService<NoopRawMutex, (), 128> = ContextService::new();
        let mut state = ();

        let runner = svc.run(&mut state);
        pin_mut!(runner);
        let _ = futures_util::poll!(&mut runner);

        {
            let dc = drop_count.clone();
            let fut = svc.call(move |_| Heavy(dc, [0xAB; 64]));
            pin_mut!(fut);
            let _ = futures_util::poll!(&mut fut);
            let _ = futures_util::poll!(&mut runner);
            // Drop caller: ack is signalled, result stays in slot.
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);

        // Runner picks up the ack and cleans up the slot.
        let _ = futures_util::poll!(&mut runner);

        // Service should work again after cleanup.
        let caller = async { svc.call(|_| 42u32).await };
        pin_mut!(caller);
        futures_util::future::select(caller, runner).await;

        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
    }

    #[futures_test::test]
    async fn try_call_immediate() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;

        // No runner yet, slot is not free.
        assert!(!svc.try_call_immediate(|s| *s += 100));

        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let _ = futures_util::poll!(&mut runner);

            // Slot is now free.
            assert!(svc.try_call_immediate(|s| *s += 1));

            // Slot is busy (job pending).
            assert!(!svc.try_call_immediate(|s| *s += 100));

            // Let the runner process the job.
            let _ = futures_util::poll!(&mut runner);
            let _ = futures_util::poll!(&mut runner);
        }

        assert_eq!(state, 1);
    }

    /// Test that if the returned value's destructor panics during cleanup
    /// (caller dropped after submission, runner drops R), the FinishGuard
    /// recovers the service: slot is freed, needs_recovery is cleared,
    /// and subsequent calls work.
    #[test]
    #[cfg(feature = "std")]
    fn destructor_panic_recovery() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("destructor panic");
            }
        }

        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        let mut state = ();

        // Start the runner, submit a job that returns PanicOnDrop,
        // then drop the caller so the runner has to clean up R.
        let result = catch_unwind(AssertUnwindSafe(|| {
            block_on(async {
                let runner = svc.run(&mut state);
                pin_mut!(runner);
                let _ = futures_util::poll!(&mut runner);

                {
                    let fut = svc.call(|_| PanicOnDrop);
                    pin_mut!(fut);
                    let _ = futures_util::poll!(&mut fut);
                    let _ = futures_util::poll!(&mut runner);
                    // Drop caller: signals ack without taking R.
                }

                // Runner receives ack, calls drop_contents() which panics.
                let _ = futures_util::poll!(&mut runner);
            });
        }));
        assert!(result.is_err(), "expected panic from destructor");

        // Service should be usable again.
        block_on(async {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let caller = async { svc.call(|_| 42u32).await };
            pin_mut!(caller);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 42),
                _ => panic!("expected caller to complete"),
            }
        });
    }

    /// Test that if the closure panics under unwinding, the caller is
    /// stuck until dropped, after which the next runner recovers.
    #[test]
    #[cfg(feature = "std")]
    fn closure_panic_recovery() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;

        // Start runner, submit a panicking closure.
        // The runner will panic during execution.
        let result = catch_unwind(AssertUnwindSafe(|| {
            block_on(async {
                let fut = svc.call(|_: &mut i32| -> i32 { panic!("closure panic") });
                pin_mut!(fut);

                let runner = svc.run(&mut state);
                pin_mut!(runner);
                // Runner initializes.
                let _ = futures_util::poll!(&mut runner);
                // Caller submits.
                let _ = futures_util::poll!(&mut fut);
                // Runner executes — closure panics.
                let _ = futures_util::poll!(&mut runner);
            });
        }));
        assert!(result.is_err(), "expected panic from closure");

        // The caller future was inside the catch_unwind and was dropped
        // during unwinding. Its Drop sent ack (phase was Submitted).

        // Service should recover: needs_recovery is true, next runner
        // waits for ack (already sent by Drop), cleans up, works again.
        block_on(async {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            let caller = async {
                svc.call(|s| {
                    *s += 1;
                    *s
                })
                .await
            };
            pin_mut!(caller);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 1),
                _ => panic!("expected caller to complete"),
            }
        });
    }

    /// Regression test: runner dropped with a pending job in the slot.
    /// The new runner must process the pending job, not mark the slot free.
    #[futures_test::test]
    async fn restart_with_pending_job() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;

        // Start and initialize the runner, then drop it.
        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            // Poll to initialize (mark_free).
            let _ = futures_util::poll!(&mut runner);
        }

        // Submit a job. The slot is free, so the caller acquires and submits.
        let caller = svc.call(|s| {
            *s += 1;
            *s
        });
        pin_mut!(caller);
        // Poll caller: acquires slot, stores F, signals job. Then poll_result → Pending.
        let _ = futures_util::poll!(&mut caller);

        // Now the slot contains F, job is signaled, but no runner is active.
        // Start a new runner. It must NOT mark_free (initialized is true).
        // It should pick up the pending job via job.wait().
        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);

            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((r, _)) => assert_eq!(r, 1),
                _ => panic!(),
            }
        }
    }
}
