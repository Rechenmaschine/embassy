#![cfg(loom)]

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use loom::sync::Arc;
use loom::thread;

/// Minimal block_on that works under loom.
/// Parks the current loom thread when pending; wakers call `unpark`.
fn block_on<F: Future>(f: F) -> F::Output {
    let thread_handle = loom::thread::current();
    let arc = Arc::new(thread_handle);

    fn clone_waker(data: *const ()) -> RawWaker {
        let arc = unsafe { Arc::from_raw(data as *const loom::thread::Thread) };
        let cloned = arc.clone();
        core::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
    }
    fn wake(data: *const ()) {
        let arc = unsafe { Arc::from_raw(data as *const loom::thread::Thread) };
        arc.unpark();
    }
    fn wake_by_ref(data: *const ()) {
        let arc = unsafe { Arc::from_raw(data as *const loom::thread::Thread) };
        arc.unpark();
        core::mem::forget(arc);
    }
    fn drop_waker(data: *const ()) {
        unsafe { Arc::from_raw(data as *const loom::thread::Thread) };
    }

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);

    let raw = RawWaker::new(Arc::into_raw(arc) as *const (), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut cx = Context::from_waker(&waker);

    let mut f = pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => loom::thread::park(),
        }
    }
}

mod channel {
    use super::*;
    use embassy_sync::blocking_mutex::raw::LoomRawMutex;
    use embassy_sync::channel::Channel;

    /// Producer on one thread does async `send`, consumer on another does async
    /// `receive`. The channel has capacity 1, so the producer *will* block if
    /// it gets ahead. Loom explores every interleaving of park/unpark between
    /// the two threads.
    #[test]
    fn async_send_receive() {
        loom::model(|| {
            let ch = Arc::new(Channel::<LoomRawMutex, u32, 1>::new());

            let ch1 = ch.clone();
            let producer = thread::spawn(move || {
                block_on(ch1.send(42));
            });

            let ch2 = ch.clone();
            let consumer = thread::spawn(move || block_on(ch2.receive()));

            producer.join().unwrap();
            let val = consumer.join().unwrap();
            assert_eq!(val, 42);
        });
    }

    /// Two producers race to send into a capacity-1 channel while a consumer
    /// drains it. One producer will have to block until the consumer makes room.
    #[test]
    fn async_two_producers_one_consumer() {
        loom::model(|| {
            let ch = Arc::new(Channel::<LoomRawMutex, u32, 1>::new());

            let ch1 = ch.clone();
            let p1 = thread::spawn(move || block_on(ch1.send(1)));

            let ch2 = ch.clone();
            let p2 = thread::spawn(move || block_on(ch2.send(2)));

            let ch3 = ch.clone();
            let consumer = thread::spawn(move || {
                let a = block_on(ch3.receive());
                let b = block_on(ch3.receive());
                let mut vals = [a, b];
                vals.sort();
                vals
            });

            p1.join().unwrap();
            p2.join().unwrap();
            let vals = consumer.join().unwrap();
            assert_eq!(vals, [1, 2]);
        });
    }

    /// Async send blocks when channel is full, then completes after the
    /// receiver drains it from another thread.
    #[test]
    fn async_send_blocks_then_unblocks() {
        loom::model(|| {
            let ch = Arc::new(Channel::<LoomRawMutex, u32, 1>::new());

            // Fill the channel
            ch.try_send(1).unwrap();

            let ch1 = ch.clone();
            let producer = thread::spawn(move || {
                // This must block until the consumer makes room
                block_on(ch1.send(2));
            });

            let ch2 = ch.clone();
            let consumer = thread::spawn(move || {
                let a = block_on(ch2.receive());
                let b = block_on(ch2.receive());
                (a, b)
            });

            producer.join().unwrap();
            let (a, b) = consumer.join().unwrap();
            assert_eq!(a, 1);
            assert_eq!(b, 2);
        });
    }
}

mod semaphore {
    use super::*;
    use embassy_sync::blocking_mutex::raw::LoomRawMutex;
    use embassy_sync::semaphore::{GreedySemaphore, Semaphore};

    /// Two threads race to async-acquire the single permit. Exactly one
    /// blocks until the other releases. Loom explores all orderings.
    #[test]
    fn async_acquire_contention() {
        loom::model(|| {
            let sem = Arc::new(GreedySemaphore::<LoomRawMutex>::new(1));

            let sem1 = sem.clone();
            let t1 = thread::spawn(move || {
                let releaser = block_on(sem1.acquire(1)).unwrap();
                // hold briefly, then drop → release
                drop(releaser);
            });

            let sem2 = sem.clone();
            let t2 = thread::spawn(move || {
                let releaser = block_on(sem2.acquire(1)).unwrap();
                drop(releaser);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // All permits are back
            assert!(sem.try_acquire(1).is_some());
        });
    }

    /// One thread releases a permit while another is waiting to acquire it.
    /// Verifies the waker logic actually unparks the blocked acquirer.
    #[test]
    fn async_acquire_waits_for_release() {
        loom::model(|| {
            let sem = Arc::new(GreedySemaphore::<LoomRawMutex>::new(0));

            let sem1 = sem.clone();
            let acquirer = thread::spawn(move || {
                let releaser = block_on(sem1.acquire(1)).unwrap();
                assert_eq!(releaser.permits(), 1);
                drop(releaser);
            });

            let sem2 = sem.clone();
            let releaser = thread::spawn(move || {
                sem2.release(1);
            });

            acquirer.join().unwrap();
            releaser.join().unwrap();

            // Permit was acquired then released, so it's back
            assert!(sem.try_acquire(1).is_some());
        });
    }

    /// Acquire 2 permits when only 1 exists, then another thread releases 1
    /// more. Tests that the waker fires correctly when the threshold is met.
    #[test]
    fn async_acquire_multiple_waits_for_enough() {
        loom::model(|| {
            let sem = Arc::new(GreedySemaphore::<LoomRawMutex>::new(1));

            let sem1 = sem.clone();
            let acquirer = thread::spawn(move || {
                let releaser = block_on(sem1.acquire(2)).unwrap();
                assert_eq!(releaser.permits(), 2);
                drop(releaser);
            });

            let sem2 = sem.clone();
            let releaser_thread = thread::spawn(move || {
                sem2.release(1);
            });

            acquirer.join().unwrap();
            releaser_thread.join().unwrap();

            assert!(sem.try_acquire(2).is_some());
        });
    }
}
