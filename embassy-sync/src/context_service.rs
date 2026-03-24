//! Execute blocking closures against `&mut T` on behalf of async callers.
//!
//! Zero-alloc. The future returned by [`ContextService::call`] is cancel-safe:
//! dropping it at any await point is safe and the service remains usable.
//!
//! The runner future returned by [`ContextService::run`] is cancel-safe and
//! restartable: dropping it at any await point is safe, and calling `run()`
//! again will recover any in-flight state and resume processing. Callers
//! that were blocked will transparently continue once a new runner starts.
//!
//! `S` is the inline storage capacity in bytes. Across all calls, both
//! the closure capture and the return type must fit within `S` bytes and
//! require no more alignment than the internal slot. Checked at compile time.
//!
//! The closure `f` passed to `call` must not unwind. Under
//! `panic = "unwind"`, a panic in `f` poisons the in-flight call: the
//! waiting caller will not receive a result, and recovery requires
//! dropping that call future before the service can make progress again.
//!
//! The choice of `M: RawMutex` must match the execution context.
//! `NoopRawMutex` is only safe within a single executor.
//! Use `CriticalSectionRawMutex` (or similar) when sharing across
//! interrupt priorities or executors.

use crate::blocking_mutex::raw::RawMutex;
use crate::semaphore::{GreedySemaphore, Semaphore, SemaphoreReleaser};
use crate::signal::Signal;
use core::cell::UnsafeCell;
use core::convert::Infallible;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll};

type RunFn<T, const S: usize> = unsafe fn(&JobStorage<S>, &mut T);

// TODO: if not included, signature even more ugly
type AcquireResult<'a, M> = Result<SemaphoreReleaser<'a, GreedySemaphore<M>>, Infallible>;

// TODO: Unwinding considerations?

/// # Safety
/// - `slot` must currently contain a live `F`.
/// - `R` must fit in the slot `size_of::<R>() <= S` and not exceed its
///   alignment `align_of::<R>() <= align_of::<JobStorage<S>>()`
/// After return, slot contains a live `R`.
unsafe fn run_job<T, R, F: FnOnce(&mut T) -> R, const S: usize>(slot: &JobStorage<S>, state: &mut T) {
    unsafe {
        let f: F = slot.take();
        let res = f(state);
        slot.store(res);
    }
}

/// Serializes closure execution against `&mut T` on a dedicated runner task.
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
/// // runner task, spawned in separate thread
/// FS.run(&mut filesystem).await;
///
/// // caller task
/// let size = FS.call(|fs| fs.read_blocking(path).len()).await;
/// ```
pub struct ContextService<M: RawMutex, T, const S: usize> {
    slot: JobStorage<S>,
    permit: GreedySemaphore<M>,
    job: Signal<M, RunFn<T, S>>,
    /// Signals to the caller that the result is ready in the slot.
    done: Signal<M, ()>,
    /// Signals to the runner the slot is free to clean up.
    ack: Signal<M, ()>,
    /// `true` if there is already a call to `run()` active.
    running: AtomicBool, // guards against concurrent runners
    /// `true` if the initial semaphore permit is released.
    initialized: AtomicBool, // guards against releasing multiple permits
    /// `true` while a job is in-flight (between receiving `job` and
    /// completing cleanup). Any subsequent `run()` checks this to recover
    /// stale state left by a cancelled runner.
    needs_recovery: AtomicBool,
}

impl<M: RawMutex, T, const S: usize> ContextService<M, T, S> {
    pub const fn new() -> Self {
        Self {
            slot: JobStorage::new(),
            permit: GreedySemaphore::new(0),
            job: Signal::new(),
            done: Signal::new(),
            ack: Signal::new(),
            running: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
            needs_recovery: AtomicBool::new(false),
        }
    }

    /// Submit a closure for execution on the runner and wait for the result.
    ///
    /// This function will err at compile time, if `F` or `R` exceeds the slot capacity `S`.
    ///
    /// ## Cancellation
    ///
    /// The returned future is cancel-safe: dropping it at any point is
    /// sound and leaves the service in a usable state. If dropped before
    /// the closure has been submitted to the runner, no work is performed.
    /// If dropped after submission, the runner will execute the closure to
    /// completion and the return value is discarded.
    // TODO somewhat ugly signature; sensible to replace return type just with impl Future?
    pub fn call<R, F>(&self, f: F) -> CallFuture<'_, M, T, R, F, impl Future<Output = AcquireResult<'_, M>>, S>
    where
        F: FnOnce(&mut T) -> R + Send + 'static,
        R: Send + 'static,
    {
        const { assert_slot_fits::<F, R, S>() };

        CallFuture {
            svc: self,
            acquire: self.permit.acquire(1),
            f: Some(f),
            phase: Phase::Acquiring,
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// Caller must have received `ack`, guaranteeing exclusive logical
    /// access to the slot.
    unsafe fn finish_job(&self) {
        unsafe { self.slot.drop_contents() };
        self.done.reset();
        self.needs_recovery.store(false, Ordering::Release);
        self.permit.release(1);
    }

    /// Run the service loop, executing closures submitted via [`call`](Self::call)
    /// with exclusive `&mut T` access.
    ///
    /// # Panics
    ///
    /// Panics if called while another runner is still active (concurrent runs).
    /// Sequential calls after a previous runner was dropped are fine.
    ///
    /// # Cancellation
    ///
    /// This future is cancel-safe. Dropping this future at any await point is sound and
    /// the service remains usable. A subsequent call to `run()` will recover any
    /// in-flight state and resume processing. Callers that were blocked will
    /// transparently continue once the new runner starts.
    pub async fn run(&self, state: &mut T) -> ! {
        if self.running.swap(true, Ordering::Acquire) {
            panic!("ContextService::run() must not be called concurrently")
        }

        struct RunGuard<'a> {
            running: &'a AtomicBool,
        }
        impl Drop for RunGuard<'_> {
            fn drop(&mut self) {
                self.running.store(false, Ordering::Release);
            }
        }

        // RunGuard clears `running` when this future is dropped,
        // allowing a subsequent run() call.
        let _guard = RunGuard { running: &self.running };

        // If the previous runner was cancelled mid-job, the caller may still
        // be interacting with the slot. Wait for it to finish (the caller
        // always acks — either explicitly or via its Drop), then clean up.
        if self.needs_recovery.load(Ordering::Acquire) {
            self.ack.wait().await;
            // SAFETY: ack was received so we have exclusive slot access.
            unsafe { self.finish_job() };
        }

        // Release the initial permit exactly once across all run() calls.
        if !self.initialized.swap(true, Ordering::Relaxed) {
            self.permit.release(1);
        }

        loop {
            // Wait for a caller to submit a type-erased closure.
            // This is a clean cancellation point — no job in flight.
            let run_fn = self.job.wait().await;

            // Mark in-flight so a subsequent run() knows to recover.
            self.needs_recovery.store(true, Ordering::Release);

            // Execute the closure.
            // SAFETY: The slot contains a live F and run_fn matches its types.
            // No other task can access the slot, since the (unique) caller is
            // waiting on done. After this, the slot contains a live R.
            unsafe { run_fn(&self.slot, state) };

            // Result is ready and can be read out. Until ack, we
            // may not access the JobStorage.
            self.done.signal(());

            // Wait for the caller to read R and signal explicitly,
            // or to signal ack on cancellation.
            // If cancelled here, needs_recovery is true and the next
            // run() will wait for ack before touching the slot.
            self.ack.wait().await;

            // The full job is complete. The caller no longer has access
            // to the store, so we can process the next job.
            // SAFETY: ack was received — exclusive slot access.
            unsafe { self.finish_job() };
        }
    }

    pub fn try_call_immediate<F>(&self, f: F) -> bool
    where
        F: FnOnce(&mut T) + Send + 'static,
    {
        const { assert_slot_fits::<F, (), S>() };

        let Some(releaser) = self.permit.try_acquire(1) else {
            return false;
        };

        releaser.disarm();

        // SAFETY:
        // - permit guarantees exclusive ownership of the slot
        // - F fits/alignment checked above
        unsafe { self.slot.store(f) };

        // No one will ever wait on `done` or read `()`, so make the runner's
        // ack wait complete immediately once it reaches that point.
        self.ack.signal(());

        self.job.signal(run_job::<T, (), F, S>);
        true
    }
}

// SAFETY: access is serialized by the call/run protocol, where
// synchronisation is handled with embassy-sync primitives.
unsafe impl<M: RawMutex, T, const S: usize> Sync for ContextService<M, T, S>
where
    GreedySemaphore<M>: Sync,
    Signal<M, RunFn<T, S>>: Sync,
    Signal<M, ()>: Sync,
{
}

enum Phase {
    /// Waiting on semaphore permit
    Acquiring,
    /// Closure is in the slot, waiting for runner to finish and write result
    Submitted,
    /// Result collected
    Done,
}

/// Future returned by [`ContextService::call`].
///
/// ## Cancellation
///
/// This future is cancel-safe: dropping it at any point is sound and
/// leaves the service in a usable state. However, behaviour differs based
/// on how much progress has been made.
///
/// If dropped before the closure has been submitted to the runner, no
/// work is performed and the closure is dropped with the future. If
/// dropped after submission, the runner will execute the closure
/// to completion and the return value is discarded.
pub struct CallFuture<'a, M: RawMutex, T, R, F, AcqFut, const S: usize> {
    svc: &'a ContextService<M, T, S>,
    acquire: AcqFut,
    f: Option<F>,
    phase: Phase,
    _marker: PhantomData<R>,
}

impl<'a, M, T, R, F, AcqFut, const S: usize> CallFuture<'a, M, T, R, F, AcqFut, S>
where
    M: RawMutex,
    F: FnOnce(&mut T) -> R + Send + 'static,
    R: Send + 'static,
    AcqFut: Future<Output = Result<SemaphoreReleaser<'a, GreedySemaphore<M>>, Infallible>>,
{
    /// Acquire the permit for exclusive access, then write the closure into
    /// the slot and wake the runner to begin execution.
    ///
    /// # Safety
    /// `self.acquire` must be in its original pinned position.
    unsafe fn poll_acquire(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        // SAFETY: `acquire` was not moved after `self` was pinned.
        let pinned = unsafe { Pin::new_unchecked(&mut self.acquire) };
        match pinned.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(releaser)) => {
                // We want to hold the permit across scopes. The runner will release
                // the permit after we signal the ack.
                releaser.disarm();

                // Write the closure into the slot and wake the runner to begin execution.
                // f is NOT structurally pinned so it is safe to move out.
                // SAFETY:
                // - slot is initially empty, and cleaned by runner after each job
                // - F fits and is aligned, ensured with compile-time checks
                let f = self.f.take().unwrap();
                unsafe { self.svc.slot.store(f) };
                self.svc.job.signal(run_job::<T, R, F, S>);

                // From here, if we're dropped, Drop signals ack which will allow
                // the runner to make progress.
                self.phase = Phase::Submitted;
                Poll::Ready(())
            }
            Poll::Ready(Err(e)) => match e {},
        }
    }

    /// Poll for the runner's result
    fn poll_result(&mut self, cx: &mut Context<'_>) -> Poll<R> {
        // TODO: recreating future here is probably ok, because we are the only consumer
        // waiting on the signal, so we won't miss updates...?
        let mut fut = self.svc.done.wait();
        match Pin::new(&mut fut).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(()) => {
                // The runner has completed the job. The return value R becomes visible
                // before the done signal is fired.

                // SAFETY: slot contains a live R written by run_job.
                let result = unsafe { self.svc.slot.take::<R>() };

                // Notify the runner we're done with the slot.
                // We must not touch the slot after this.
                self.svc.ack.signal(());
                self.phase = Phase::Done;
                Poll::Ready(result)
            }
        }
    }
}

impl<'a, M, T, R, F, AcqFut, const S: usize> Future for CallFuture<'a, M, T, R, F, AcqFut, S>
where
    M: RawMutex,
    F: FnOnce(&mut T) -> R + Send + 'static,
    R: Send + 'static,
    AcqFut: Future<Output = Result<SemaphoreReleaser<'a, GreedySemaphore<M>>, Infallible>>,
{
    type Output = R;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // SAFETY: once `self` is pinned, `acquire` is never moved.
        // `f` is not pinned and may be moved out.
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match this.phase {
                Phase::Acquiring => match unsafe { this.poll_acquire(cx) } {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(()) => {} // fall through to poll_result
                },
                Phase::Submitted => return this.poll_result(cx),
                Phase::Done => panic!("`CallFut` polled after completion"),
            }
        }
    }
}

impl<M: RawMutex, T, R, F, AcqFut, const S: usize> Drop for CallFuture<'_, M, T, R, F, AcqFut, S> {
    fn drop(&mut self) {
        if matches!(self.phase, Phase::Submitted) {
            // Future dropped after the job was submitted.
            // This does NOT cancel the job, the runner will still finish executing it.
            //
            // We must not touch the slot here, since the runner may still be
            // reading `F`, executing it, or writing back `R`.
            //
            // We also can't wait for it, because `Drop` should not block.
            // So we just signal the runner. Cleanup is deferred to the runner.
            self.svc.ack.signal(());
        }
    }
}

/// Type-erased storage for jobs. Invariants:
/// - `drop_fn = Some(drop_glue::<T>)` iff the slot contains a live `T`
/// - `store<T>()` writes a `T` and arms `drop_fn`
/// - `take<T>()` reads a `T` out and clears `drop_fn`
/// - `drop_contents()` drops in place if occupied; no-op if empty
// TODO: Is ≥8-aligned ok?
#[repr(C, align(8))]
struct JobStorage<const S: usize> {
    buf: UnsafeCell<MaybeUninit<[u8; S]>>,
    // erased instance of the drop_glue::<T>() function
    drop_fn: UnsafeCell<Option<unsafe fn(&Self)>>,
}

impl<const S: usize> JobStorage<S> {
    const fn new() -> Self {
        Self {
            buf: UnsafeCell::new(MaybeUninit::uninit()),
            drop_fn: UnsafeCell::new(None),
        }
    }

    /// # Safety
    /// Slot must be empty. `T` must fit s.t `size_of::<T>() ≤ S` bytes and `align_of::<T>() ≤ align_of::<Self>()`
    unsafe fn store<T>(&self, val: T) {
        unsafe {
            ((*self.buf.get()).as_mut_ptr() as *mut T).write(val);
            *self.drop_fn.get() = Some(Self::drop_glue::<T>);
        }
    }

    /// # Safety
    /// Slot must contain a live `T`.
    unsafe fn take<T>(&self) -> T {
        unsafe {
            let val = ((*self.buf.get()).as_ptr() as *const T).read();
            *self.drop_fn.get() = None;
            val
        }
    }

    /// Drop the stored value (if any) and clear
    ///
    /// # Safety
    /// Caller must have exclusive access
    unsafe fn drop_contents(&self) {
        unsafe {
            let f = (*self.drop_fn.get()).take();
            if let Some(f) = f {
                f(self);
            }
        }
    }

    /// Drop the value currently stored in the slot as a `T`
    ///
    /// # Safety
    /// `slot` must currently contain a live `T`.
    unsafe fn drop_glue<T>(slot: &Self) {
        unsafe {
            core::ptr::drop_in_place((*slot.buf.get()).as_mut_ptr() as *mut T);
        }
    }
}

impl<const S: usize> Drop for JobStorage<S> {
    fn drop(&mut self) {
        // SAFETY: `&mut self` guarantees exclusive access.
        unsafe { self.drop_contents() };
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
        mem::align_of::<F>() <= mem::align_of::<JobStorage<S>>(),
        "closure alignment must not exceed 8 bytes"
    );
    assert!(
        mem::align_of::<R>() <= mem::align_of::<JobStorage<S>>(),
        "return type alignment must not exceed 8 bytes"
    );
}

#[cfg(all(test))]
mod tests {
    extern crate alloc;

    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::AtomicUsize;

    use super::*;
    use crate::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex};
    use futures_executor::block_on;
    use futures_util::pin_mut;

    #[futures_test::test]
    async fn basic() {
        let svc: ContextService<NoopRawMutex, i32, 2> = ContextService::new();
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
    async fn cancel_before_send() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let _fut = svc.call(|s| {
            *s += 1;
            *s
        });
    }

    #[futures_test::test]
    async fn cancel_before_permit_then_next_call() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            {
                let fut = svc.call(|s: &mut i32| {
                    *s += 100;
                    *s
                });
                futures_util::pin_mut!(fut);
                assert!(futures_util::poll!(&mut fut).is_pending());
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
    async fn closure_dropped_on_cancel_before_permit() {
        use core::sync::atomic::AtomicUsize;

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
            futures_util::pin_mut!(fut);
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

        // First run: do one call, then drop the runner.
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

        // Second run: service should work again.
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
        use core::sync::atomic::AtomicUsize;

        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        let mut state = ();

        let drop_count = Arc::new(AtomicUsize::new(0));

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Start runner, submit a job that returns a Tracked value,
        // then cancel the caller (so it acks without reading R).
        // Then cancel the runner (mid-job: after done, waiting for ack).
        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            // Poll runner to initialize it.
            let _ = futures_util::poll!(&mut runner);

            {
                let dc = drop_count.clone();
                let fut = svc.call(move |_| Tracked(dc));
                pin_mut!(fut);
                // Poll caller to submit the job.
                let _ = futures_util::poll!(&mut fut);
                // Poll runner to execute the job and signal done.
                let _ = futures_util::poll!(&mut runner);
                // Drop caller — its Drop signals ack.
            }
            // Runner is dropped here while needs_recovery is true.
        }

        // The Tracked return value should not have been dropped yet
        // (it's in the slot, waiting for recovery).
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);

        // Restart the runner. Recovery should clean up the slot.
        {
            let runner = svc.run(&mut state);
            pin_mut!(runner);
            // Poll to trigger recovery.
            let _ = futures_util::poll!(&mut runner);

            assert_eq!(drop_count.load(Ordering::Relaxed), 1);

            // Service should work normally after recovery.
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
    async fn closure_dropped_on_cancel() {
        use core::sync::atomic::AtomicUsize;

        let drop_count = Arc::new(AtomicUsize::new(0));
        let dc = drop_count.clone();

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        {
            let tracked = Tracked(dc);
            let _fut = svc.call(move |_| {
                let _ = &tracked;
            });
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
    }

    #[futures_test::test]
    async fn miri_store_take_roundtrip() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 42i32;
        let caller = async {
            let r: i32 = svc
                .call(|s| {
                    let v = *s;
                    *s = 0;
                    v
                })
                .await;
            assert_eq!(r, 42);
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        match futures_util::future::select(caller, runner).await {
            futures_util::future::Either::Left(((), _)) => {}
            _ => panic!(),
        }
    }

    #[futures_test::test]
    async fn miri_drop_glue_on_service_drop() {
        use core::sync::atomic::AtomicUsize;

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
    async fn miri_heterogeneous_types_no_ub() {
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
    async fn miri_cancel_after_submit_no_leak() {
        use core::sync::atomic::AtomicUsize;

        let drop_count = Arc::new(AtomicUsize::new(0));

        struct Heavy(Arc<AtomicUsize>, [u8; 64]);
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
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);

        let _ = futures_util::poll!(&mut runner);

        let caller = async { svc.call(|_| 42u32).await };
        pin_mut!(caller);
        futures_util::future::select(caller, runner).await;

        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
    }

    #[futures_test::test]
    async fn miri_zst_closure_and_return() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            svc.call(|s| {
                *s = 99;
            })
            .await;
            assert_eq!(svc.call(|s| *s).await, 99);
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn miri_aligned_u64_in_slot() {
        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        let mut state = ();
        let caller = async {
            let v: u64 = svc.call(|_| 0xDEAD_BEEF_CAFE_BABEu64).await;
            assert_eq!(v, 0xDEAD_BEEF_CAFE_BABE);
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[futures_test::test]
    async fn miri_multiple_cancels_then_success() {
        let svc: ContextService<NoopRawMutex, i32, 64> = ContextService::new();
        let mut state = 0i32;
        let caller = async {
            for _ in 0..3 {
                let fut = svc.call(|s: &mut i32| {
                    *s += 100;
                    *s
                });
                pin_mut!(fut);
                assert!(futures_util::poll!(&mut fut).is_pending());
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
    async fn miri_box_in_slot() {
        let svc: ContextService<NoopRawMutex, (), 64> = ContextService::new();
        let mut state = ();
        let caller = async {
            let b: Box<[u8]> = svc.call(|_| vec![1, 2, 3].into_boxed_slice()).await;
            assert_eq!(&*b, &[1, 2, 3]);
        };
        let runner = svc.run(&mut state);
        pin_mut!(caller);
        pin_mut!(runner);
        futures_util::future::select(caller, runner).await;
    }

    #[test]
    fn threaded_sequential_20() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        let pool = ThreadPool::new().unwrap();
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, i32, 64>::new());

        let svc_c = svc.clone();
        let caller = pool
            .spawn_with_handle(async move {
                let mut results = vec![];
                for _ in 0..20 {
                    results.push(
                        svc_c
                            .call(|s| {
                                *s += 1;
                                *s
                            })
                            .await,
                    );
                }
                results
            })
            .unwrap();

        block_on(async {
            let mut state = 0i32;
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((results, _)) => {
                    assert_eq!(results, (1..=20).collect::<Vec<_>>());
                }
                _ => panic!(),
            }
        });
    }

    #[test]
    fn threaded_cancel_and_retry() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        let pool = ThreadPool::new().unwrap();
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, i32, 64>::new());

        let svc_c = svc.clone();
        let caller = pool
            .spawn_with_handle(async move {
                for _ in 0..5 {
                    let fut = svc_c.call(|s: &mut i32| {
                        *s += 100;
                        *s
                    });
                    futures_util::pin_mut!(fut);
                    let _ = futures_util::poll!(&mut fut);
                }
                svc_c
                    .call(|s| {
                        *s += 1;
                        *s
                    })
                    .await
            })
            .unwrap();

        block_on(async {
            let mut state = 0i32;
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            match futures_util::future::select(caller, runner).await {
                futures_util::future::Either::Left((result, _)) => {
                    assert!(result > 0);
                }
                _ => panic!(),
            }
        });
    }

    #[test]
    fn threaded_different_return_types() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        let pool = ThreadPool::new().unwrap();
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, Vec<String>, 256>::new());

        let svc_c = svc.clone();
        let caller = pool
            .spawn_with_handle(async move {
                svc_c.call(|s: &mut Vec<String>| s.push("hello".into())).await;
                let len: usize = svc_c.call(|s: &mut Vec<String>| s.len()).await;
                assert_eq!(len, 1);
                let joined: String = svc_c.call(|s: &mut Vec<String>| s.join(", ")).await;
                assert_eq!(joined, "hello");
            })
            .unwrap();

        block_on(async {
            let mut state = Vec::new();
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            futures_util::future::select(caller, runner).await;
        });
    }

    #[test]
    fn threaded_drop_tracking() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let pool = ThreadPool::new().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, (), 64>::new());

        let svc_c = svc.clone();
        let d = drops.clone();
        let caller = pool
            .spawn_with_handle(async move {
                for _ in 0..5 {
                    let dc = d.clone();
                    let t = svc_c.call(move |_| Tracked(dc)).await;
                    drop(t);
                }
            })
            .unwrap();

        block_on(async {
            let mut state = ();
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            futures_util::future::select(caller, runner).await;
        });

        assert_eq!(drops.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn threaded_stress_100() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        let pool = ThreadPool::new().unwrap();
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, u64, 64>::new());

        let svc_c = svc.clone();
        let caller = pool
            .spawn_with_handle(async move {
                for i in 1..=100u64 {
                    assert_eq!(
                        svc_c
                            .call(|s| {
                                *s += 1;
                                *s
                            })
                            .await,
                        i
                    );
                }
            })
            .unwrap();

        block_on(async {
            let mut state = 0u64;
            let runner = svc.run(&mut state);
            pin_mut!(caller);
            pin_mut!(runner);
            futures_util::future::select(caller, runner).await;
        });
    }

    #[test]
    fn threaded_two_callers() {
        use futures_executor::ThreadPool;
        use futures_util::task::SpawnExt;

        let pool = ThreadPool::new().unwrap();
        let svc = Arc::new(ContextService::<CriticalSectionRawMutex, i32, 64>::new());

        let svc_a = svc.clone();
        let a = pool
            .spawn_with_handle(async move {
                let mut r: Vec<i32> = vec![];
                for _ in 0..10 {
                    r.push(
                        svc_a
                            .call(|s: &mut i32| {
                                *s += 1;
                                *s
                            })
                            .await,
                    );
                }
                r
            })
            .unwrap();

        let svc_b = svc.clone();
        let b = pool
            .spawn_with_handle(async move {
                let mut r: Vec<i32> = vec![];
                for _ in 0..10 {
                    r.push(
                        svc_b
                            .call(|s: &mut i32| {
                                *s += 1;
                                *s
                            })
                            .await,
                    );
                }
                r
            })
            .unwrap();

        block_on(async {
            let mut state = 0i32;
            let runner = svc.run(&mut state);
            let callers = futures_util::future::join(a, b);
            pin_mut!(callers);
            pin_mut!(runner);
            match futures_util::future::select(callers, runner).await {
                futures_util::future::Either::Left(((ra, rb), _)) => {
                    let mut all: Vec<i32> = ra.into_iter().chain(rb).collect();
                    all.sort();
                    assert_eq!(all, (1..=20).collect::<Vec<_>>());
                }
                _ => panic!(),
            }
        });
    }
}
