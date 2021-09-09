//! A futures-aware [`RwLock`] with `ref_count`, `upgrade`, and `downgrade` methods.
//!
//! Based on the `RwLock` in [`futures_locks`](https://crates.io/crates/futures-locks),
//! which in turn is based on [`std::sync::RwLock`].
//!
//! Does not use message passing.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use futures::task::{Context, Poll, Waker};

/// A read guard for [`RwLock`] (can be dereferenced into `&T`)
pub struct RwLockReadGuard<T> {
    lock: RwLock<T>,
}

impl<T> RwLockReadGuard<T> {
    /// Upgrade this read lock in to a write lock.
    pub fn upgrade(self) -> RwLockWriteFuture<T> {
        self.lock.write()
    }
}

impl<T> Clone for RwLockReadGuard<T> {
    fn clone(&self) -> RwLockReadGuard<T> {
        let mut state = self.lock.inner.state.lock().expect("RwLockReadGuard state");
        state.readers += 1;

        RwLockReadGuard {
            lock: self.lock.clone(),
        }
    }
}

impl<T> Deref for RwLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self
                .lock
                .inner
                .state
                .lock()
                .expect("RwLockReadGuard read")
                .value
                .get()
        }
    }
}

impl<T> Drop for RwLockReadGuard<T> {
    fn drop(&mut self) {
        let mut state = self
            .lock
            .inner
            .state
            .lock()
            .expect("RwLockReadGuard drop state");

        state.readers -= 1;
        state.wake();
    }
}

/// A write guard for [`RwLock`] (can be dereferenced into `&mut T`)
pub struct RwLockWriteGuard<T> {
    lock: RwLock<T>,
}

impl<T> RwLockWriteGuard<T> {
    /// Downgrade this write lock into a read lock.
    pub fn downgrade(self) -> RwLockReadFuture<T> {
        self.lock.read()
    }
}

impl<T> Deref for RwLockWriteGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self
                .lock
                .inner
                .state
                .lock()
                .expect("RwLockWriteGuard read")
                .value
                .get()
        }
    }
}

impl<T> DerefMut for RwLockWriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            &mut *self
                .lock
                .inner
                .state
                .lock()
                .expect("RwLockWriteGuard write")
                .value
                .get()
        }
    }
}

impl<T> Drop for RwLockWriteGuard<T> {
    fn drop(&mut self) {
        let mut state = self
            .lock
            .inner
            .state
            .lock()
            .expect("RwLockWriteGuard drop state");

        state.writer = false;
        state.wake();
    }
}

struct LockState<T> {
    readers: usize,
    writer: bool,
    next_id: usize,
    wakers: HashMap<usize, Waker>,
    value: UnsafeCell<T>,
}

impl<T> LockState<T> {
    fn consume_id(&mut self) -> usize {
        let id = self.next_id;

        self.next_id = if let Some(next) = self.next_id.checked_add(1) {
            next
        } else {
            0
        };

        id
    }

    fn wake(&mut self) {
        for (_, waker) in self.wakers.drain() {
            waker.wake();
        }
    }
}

struct Inner<T> {
    state: Mutex<LockState<T>>,
}

pub struct RwLock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for RwLock<T> {
    fn clone(&self) -> RwLock<T> {
        RwLock {
            inner: self.inner.clone(),
        }
    }
}

/// A futures-aware version of [`std::sync::RwLock`].
impl<T> RwLock<T> {
    /// Construct a new `RwLock` with the given `value`.
    pub fn new(value: T) -> RwLock<T> {
        let state = LockState {
            readers: 0,
            writer: false,
            next_id: 0,
            wakers: HashMap::new(),
            value: UnsafeCell::new(value),
        };

        let inner = Inner {
            state: Mutex::new(state),
        };

        RwLock {
            inner: Arc::new(inner),
        }
    }

    /// Return a read lock synchronously if possible, otherwise `None`.
    pub fn try_read(&self) -> Option<RwLockReadGuard<T>> {
        if let Ok(mut state) = self.inner.state.try_lock() {
            if state.writer {
                None
            } else {
                state.readers += 1;
                Some(RwLockReadGuard { lock: self.clone() })
            }
        } else {
            None
        }
    }

    /// Return a read lock asynchronously.
    pub fn read(&self) -> RwLockReadFuture<T> {
        let mut state = self.inner.state.lock().expect("RwLock::read");
        let id = state.consume_id();
        RwLockReadFuture::new(id, self.clone())
    }

    /// Return a write lock synchronously if possible, otherwise `None`.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<T>> {
        if let Ok(mut state) = self.inner.state.try_lock() {
            if state.writer || state.readers > 0 {
                None
            } else {
                state.writer = true;
                Some(RwLockWriteGuard { lock: self.clone() })
            }
        } else {
            None
        }
    }

    /// Return a write lock asynchronously.
    pub fn write(&self) -> RwLockWriteFuture<T> {
        let mut state = self.inner.state.lock().expect("RwLock::write");
        let id = state.consume_id();
        RwLockWriteFuture::new(id, self.clone())
    }

    /// Return the current number of references to this `RwLock`.
    ///
    /// Note that it is possible for this value to change at any time, including between
    /// calling `ref_count` and acting on the result.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

/// A [`Future`] representing a pending read lock.
pub struct RwLockReadFuture<T> {
    id: usize,
    lock: RwLock<T>,
}

impl<T> RwLockReadFuture<T> {
    fn new(id: usize, lock: RwLock<T>) -> Self {
        Self { id, lock }
    }
}

impl<T> Future for RwLockReadFuture<T> {
    type Output = RwLockReadGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_read() {
            Some(guard) => Poll::Ready(guard),
            None => {
                let mut state = self.lock.inner.state.lock().unwrap();
                state.wakers.insert(self.id, context.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// A [`Future`] representing a pending write lock.
pub struct RwLockWriteFuture<T> {
    id: usize,
    lock: RwLock<T>,
}

impl<T> RwLockWriteFuture<T> {
    fn new(id: usize, lock: RwLock<T>) -> Self {
        Self { id, lock }
    }
}

impl<T> Future for RwLockWriteFuture<T> {
    type Output = RwLockWriteGuard<T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        match self.lock.try_write() {
            Some(guard) => Poll::Ready(guard),
            None => {
                let mut state = self.lock.inner.state.lock().unwrap();
                state.wakers.insert(self.id, context.waker().clone());
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn my_test() {
        let n = RwLock::new(5u32);
        let mut write_lock = n.write().await;
        assert_eq!(*write_lock, 5);
        assert!(n.try_read().is_none());

        *write_lock = 6;

        {
            let read_lock = write_lock.downgrade().await;
            assert_eq!(*read_lock, 6);

            let second_read_lock = n.read().await;
            assert_eq!(*second_read_lock, 6);

            assert!(n.try_write().is_none());
        }

        assert!(n.try_write().is_some());
    }
}
