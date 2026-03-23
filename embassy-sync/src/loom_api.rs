//! Compatibility shim for loom testing.
//!
//! Under normal builds, re-exports from `core`. Under `cfg(loom)`,
//! provides loom-instrumented equivalents so that loom can track
//! every access to shared mutable state.

pub(crate) mod cell {
    /// An `UnsafeCell` wrapper that delegates to `loom::cell::UnsafeCell`
    /// under `cfg(loom)` and to `core::cell::UnsafeCell` otherwise.
    ///
    /// Instead of `.get()`, callers use the callback-style
    /// `.with(|*const T| …)` / `.with_mut(|*mut T| …)` API.
    /// Under loom, these register read/write accesses so loom can
    /// detect data races.

    #[cfg(not(loom))]
    #[derive(Debug)]
    pub(crate) struct UnsafeCell<T: ?Sized>(core::cell::UnsafeCell<T>);

    #[cfg(not(loom))]
    impl<T> UnsafeCell<T> {
        #[inline]
        pub const fn new(data: T) -> Self {
            Self(core::cell::UnsafeCell::new(data))
        }

        #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    #[cfg(not(loom))]
    impl<T: ?Sized> UnsafeCell<T> {
        #[inline]
        pub fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
            f(self.0.get() as *const T)
        }

        #[inline]
        pub fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }

        #[inline]
        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut()
        }
    }

    #[cfg(loom)]
    #[derive(Debug)]
    pub(crate) struct UnsafeCell<T: ?Sized>(loom::cell::UnsafeCell<T>);

    #[cfg(loom)]
    impl<T> UnsafeCell<T> {
        #[inline]
        pub fn new(data: T) -> Self {
            Self(loom::cell::UnsafeCell::new(data))
        }

        #[inline]
        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    #[cfg(loom)]
    impl<T: ?Sized> UnsafeCell<T> {
        #[inline]
        pub fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
            self.0.with(f)
        }

        #[inline]
        pub fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            self.0.with_mut(f)
        }

        #[inline]
        pub fn get_mut(&mut self) -> &mut T {
            self.0.with_mut(|ptr| unsafe { &mut *ptr })
        }
    }
}

pub(crate) mod sync {
    pub(crate) mod atomic {
        #[cfg(not(loom))]
        pub(crate) use core::sync::atomic::{AtomicBool, Ordering};

        #[cfg(loom)]
        pub(crate) use loom::sync::atomic::{AtomicBool, Ordering};
    }
}
