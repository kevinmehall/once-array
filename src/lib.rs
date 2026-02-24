//! A single-producer multiple-consumer append-only fixed capacity array.
//!
//! Creating a `OnceArrayWriter<T>` allocates a fixed-capacity buffer and
//! represents exclusive access to append elements. Any number of
//! `Arc<OnceArray<T>>` references can be created and shared across threads.
//! These readers can access the slice of committed elements, and see new
//! elements as they are committed by the writer without any locking.
//!
//! `OnceArray` serves as a building block for streaming data to multiple
//! consumers while amortizing the cost of allocation and synchronization across
//! chunks of many elements.
//!
//! # Example:
//!
//! ```rust
//! use once_array::{OnceArrayWriter, OnceArray};
//! let mut writer = OnceArrayWriter::with_capacity(1024);
//!
//! // Clone the reader to share it across threads.
//! let reader1 = writer.reader().clone();
//! let reader2 = writer.reader().clone();
//!
//! // Append some data to the writer.
//! writer.try_push(42).unwrap();
//! writer.try_push(43).unwrap();
//!
//! // Commit the new elements to make them visible to readers.
//! writer.commit();
//!
//! assert_eq!(reader1.as_slice(), &[42, 43]);
//! assert_eq!(reader2.as_slice(), &[42, 43]);
//! ```

#![no_std]

extern crate alloc;

use alloc::{sync::Arc, vec::Vec};
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{ptr, slice};

/// The reader side of a single-producer multiple-consumer append-only fixed capacity array.
///
/// A `OnceArray` is normally behind `Arc` and constructed by creating a
/// [`OnceArrayWriter`] and then cloning its `.reader()`.
///
/// An owned `OnceArray<T>` is semantically identical to a `Vec<T>` but without methods to mutate it.
/// They can be inter-converted with `From` and `Into`, which may be useful in cases like:
///   * Constructing a `OnceArray` from a `Vec` populated upfront, to pass to an API that requires `OnceArray`.
///   * Unwrapping the underlying `Vec` after after claiming ownership with [`Arc::into_inner`] or [`Arc::try_unwrap`].
pub struct OnceArray<T> {
    // safety invariants:
    // * `data` and `cap` may not change
    // * `len` may never decrease
    // * `len` is always less than or equal to `cap`
    // * the first `len` elements of `data` are initialized
    // * nothing may write to or invalidate `*data..*data.add(len)`, because
    //   another thread may have a reference to it
    data: *mut T,
    len: AtomicUsize,
    cap: usize,
}

unsafe impl<T> Send for OnceArray<T> where T: Send {}
unsafe impl<T> Sync for OnceArray<T> where T: Sync {}

impl<T> Drop for OnceArray<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY:
            // * We have exclusive access guaranteed by &mut.
            // * `self.data` and `self.cap` came from a Vec,
            //    so can be turned back into a Vec.
            // * `self.len` elements are properly initialized
            drop(Vec::from_raw_parts(
                self.data,
                *self.len.get_mut(),
                self.cap,
            ))
        }
    }
}

impl<T> OnceArray<T> {
    fn from_vec(v: Vec<T>) -> Self {
        let mut v = ManuallyDrop::new(v);
        OnceArray {
            data: v.as_mut_ptr(),
            cap: v.capacity(),
            len: AtomicUsize::new(v.len()),
        }
    }

    fn into_vec(self) -> Vec<T> {
        unsafe {
            // SAFETY:
            // * We have exclusive access guaranteed by self.
            // * `self.data` and `self.cap` came from a Vec,
            //    so can be turned back into a Vec.
            // * `self.len` elements are properly initialized
            let mut v = ManuallyDrop::new(self);
            Vec::from_raw_parts(v.data, *v.len.get_mut(), v.cap)
        }
    }

    /// Returns the maximum number of elements this buffer can hold.
    ///
    /// The capacity can't change once allocated.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns the current number of elements in the buffer.
    ///
    /// This increases when the [`OnceArrayWriter`] commits new elements, but
    /// can never decrease.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the buffer is at full capacity.
    pub fn is_full(&self) -> bool {
        self.len() == self.cap
    }

    /// Obtain a slice of the committed part of the buffer.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            // SAFETY: This came from a vector and is properly aligned.
            // The part up to len is initialized, and won't change
            slice::from_raw_parts(self.data, self.len())
        }
    }
}

impl<T> Deref for OnceArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for OnceArray<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> core::borrow::Borrow<[T]> for OnceArray<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> From<Vec<T>> for OnceArray<T> {
    fn from(val: Vec<T>) -> Self {
        OnceArray::from_vec(val)
    }
}

impl<T> From<OnceArray<T>> for Vec<T> {
    fn from(v: OnceArray<T>) -> Self {
        v.into_vec()
    }
}

/// Exclusive write access to a [`OnceArray`].
///
/// The `OnceArrayWriter` provides methods to append elements to the uncommitted
/// portion of the array. The uncommitted portion is not visible to readers and
/// can be [mutated](OnceArrayWriter::uncommitted_mut) or
/// [discarded](OnceArrayWriter::revert) because the writer retains exclusive access.
///
/// Once the writer is ready to make new elements visible to readers, it can
/// call [`commit()`](OnceArrayWriter::commit) or
/// [`commit_partial(n)`](OnceArrayWriter::commit_partial) to make elements
/// immutable and atomically visible to readers. As long as there is
/// remaining capacity, the writer can continue to append and commit more elements.
///
/// The API is optimized for scenarios where data is written to a series of new
/// `OnceArrayWriter` chunks as they fill, so the append APIs return a `Result`
/// handing back the unconsumed data when full, so the caller can easily continue
/// filling the next chunk.
pub struct OnceArrayWriter<T> {
    // safety invariants:
    // * This is the only `OnceArrayWriter` that wraps `inner`.
    // * `uncommitted_len` is greater than or equal to `inner.len`, and less than or equal to `inner.cap`.
    // * `uncommitted_len` elements have been initialized.
    inner: Arc<OnceArray<T>>,
    uncommitted_len: usize,
}

impl<T> OnceArrayWriter<T> {
    fn from_vec(v: Vec<T>) -> OnceArrayWriter<T> {
        Self {
            uncommitted_len: v.len(),
            inner: Arc::new(OnceArray::from_vec(v)),
        }
    }

    /// Creates a new `OnceArrayWriter` with the specified capacity.
    pub fn with_capacity(n: usize) -> OnceArrayWriter<T> {
        Self::from_vec(Vec::with_capacity(n))
    }

    /// Obtain a read-only reference to the committed part of the array.
    pub fn reader(&self) -> &Arc<OnceArray<T>> {
        &self.inner
    }

    /// Returns the number of additional elements that can be written to the buffer before it is full.
    pub fn remaining_capacity(&self) -> usize {
        self.inner.cap - self.uncommitted_len
    }

    /// Obtain an immutable slice of the entire array, including committed and uncommitted parts.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            // SAFETY:
            // * the array has been initialized up to uncommitted_len
            slice::from_raw_parts(self.inner.data, self.uncommitted_len)
        }
    }

    /// Obtain a mutable slice of the uncommitted part of the array.
    pub fn uncommitted_mut(&mut self) -> &mut [T] {
        // SAFETY:
        // * this is above the committed len, so these elements are not shared.
        // * this is below the uncommitted_len, so these elements have been initialized.
        unsafe {
            let committed_len = self.inner.len.load(Ordering::Relaxed);
            slice::from_raw_parts_mut(
                self.inner.data.add(committed_len),
                self.uncommitted_len - committed_len,
            )
        }
    }

    unsafe fn push_unchecked(&mut self, val: T) {
        // SAFETY:
        // * caller must ensure that uncommitted_len is less than capacity
        // * uncommitted_len is greater than or equal to inner.len, so this doesn't invalidate shared slices
        // * this has &mut exclusive access to the only `OnceArrayWriter`
        //   wrapping `inner`, so no other thread is writing.
        unsafe {
            self.inner.data.add(self.uncommitted_len).write(val);
            self.uncommitted_len += 1;
        }
    }

    /// Attempts to append an element to the buffer.
    ///
    /// If the buffer is full, returns `Err(val)`, returning ownership of the value
    /// that could not be added.
    ///
    /// The new element is not visible to readers until a call to `commit()`.
    pub fn try_push(&mut self, val: T) -> Result<(), T> {
        if self.uncommitted_len < self.inner.cap {
            // SAFETY: checked that uncommitted_len is less than capacity
            unsafe {
                self.push_unchecked(val);
            }
            Ok(())
        } else {
            Err(val)
        }
    }

    /// Attempts to append elements from `iter` to the buffer.
    ///
    /// If the buffer becomes full before `iter` is exhausted, returns
    /// `Err(iter)`, returning ownership of the iterator.
    ///
    /// Note that if the iterator exactly fills the remaining capacity, this
    /// will return `Err` with an empty iterator, since the `Iterator` trait
    /// does not allow checking if an iterator is exhausted without calling
    /// `next()`.
    ///
    /// The new elements are not visible to readers until a call to `commit()`.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), I::IntoIter> {
        let mut iter = iter.into_iter();
        while self.uncommitted_len < self.inner.cap {
            if let Some(val) = iter.next() {
                // SAFETY: checked that uncommitted_len is less than capacity
                unsafe {
                    self.push_unchecked(val);
                }
            } else {
                return Ok(());
            }
        }
        Err(iter)
    }

    /// Attempts to append elements from `src` to the array.
    ///
    /// Returns the tail of the slice that could not be written to the buffer.
    /// If the buffer is not filled and all elements were written, this will be
    /// an empty slice.
    ///
    /// The new elements are not visible to readers until a call to `commit()`.
    pub fn extend_from_slice<'a>(&mut self, src: &'a [T]) -> &'a [T]
    where
        T: Copy,
    {
        let count = self.remaining_capacity().min(src.len());
        unsafe {
            // SAFETY:
            // * checked that position is less than capacity so
            //   address is in bounds.
            // * this is above the current len so doesn't invalidate slices
            // * this has &mut exclusive access to the only `OnceArrayWriter`
            //   wrapping `inner`, so no other thread is writing.
            self.inner
                .data
                .add(self.uncommitted_len)
                .copy_from_nonoverlapping(src.as_ptr(), count);
        }

        self.uncommitted_len += count;
        &src[count..]
    }

    /// Makes newly written elements immutable and atomically visible to readers.
    pub fn commit(&mut self) {
        self.inner
            .len
            .store(self.uncommitted_len, Ordering::Release);
    }

    /// Makes the first `n` newly written elements immutable and atomically visible to readers.
    ///
    /// **Panics** if `n` is greater than the number of initialized but uncommitted elements.
    pub fn commit_partial(&mut self, n: usize) {
        let committed_len = self.inner.len.load(Ordering::Relaxed);
        assert!(
            n <= self.uncommitted_len - committed_len,
            "Cannot commit more elements than have been initialized"
        );
        self.inner.len.store(committed_len + n, Ordering::Release);
    }

    /// Discards any uncommitted elements, reverting the buffer to the last committed state.
    pub fn revert(&mut self) {
        let committed_len = self.inner.len.load(Ordering::Relaxed);
        let uncommitted_len = self.uncommitted_len;

        // truncate first, in case dropping an element panics
        self.uncommitted_len = committed_len;

        // SAFETY:
        // These elements have been initialized and are not shared.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.inner.data.add(committed_len),
                uncommitted_len - committed_len,
            ));
        }
    }
}

impl<T> Drop for OnceArrayWriter<T> {
    fn drop(&mut self) {
        self.revert();
    }
}

impl<T> From<Vec<T>> for OnceArrayWriter<T> {
    fn from(vec: Vec<T>) -> OnceArrayWriter<T> {
        OnceArrayWriter::from_vec(vec)
    }
}

#[test]
fn test_to_from_vec() {
    let v = OnceArray::from(alloc::vec![1, 2, 3]);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
    let v = Vec::from(v);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn test_push() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader().clone();
    assert_eq!(reader.capacity(), 4);
    assert_eq!(reader.len(), 0);

    assert_eq!(writer.try_push(1), Ok(()));
    assert_eq!(reader.len(), 0);
    writer.commit();
    assert_eq!(reader.len(), 1);
    assert_eq!(reader.as_slice(), &[1]);

    assert_eq!(writer.try_push(2), Ok(()));
    assert_eq!(writer.try_push(3), Ok(()));
    assert_eq!(writer.try_push(4), Ok(()));
    assert_eq!(writer.try_push(5), Err(5));
    writer.commit();

    assert_eq!(reader.len(), 4);
    assert_eq!(reader.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_extend_from_slice() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader().clone();
    assert_eq!(reader.capacity(), 4);
    assert_eq!(reader.len(), 0);

    assert_eq!(writer.extend_from_slice(&[1, 2]), &[]);
    assert_eq!(reader.len(), 0);
    writer.commit();
    assert_eq!(reader.len(), 2);
    assert_eq!(reader.as_slice(), &[1, 2]);

    assert_eq!(writer.extend_from_slice(&[3, 4, 5, 6]), &[5, 6]);
    writer.commit();
    assert_eq!(reader.len(), 4);
    assert_eq!(reader.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_commit_revert() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader().clone();

    assert_eq!(writer.try_push(1), Ok(()));
    assert_eq!(writer.try_push(2), Ok(()));
    assert_eq!(writer.as_slice(), &[1, 2]);
    assert_eq!(writer.uncommitted_mut(), &mut [1, 2]);
    writer.commit();
    assert_eq!(reader.as_slice(), &[1, 2]);
    assert_eq!(writer.uncommitted_mut(), &mut []);

    assert_eq!(writer.try_push(3), Ok(()));
    assert_eq!(writer.try_push(4), Ok(()));

    writer.revert();
    assert_eq!(reader.as_slice(), &[1, 2]);
    assert_eq!(writer.uncommitted_mut(), &mut []);

    assert_eq!(writer.try_push(5), Ok(()));
    assert_eq!(writer.try_push(6), Ok(()));
    assert_eq!(writer.as_slice(), &[1, 2, 5, 6]);

    writer.commit_partial(1);
    assert_eq!(reader.as_slice(), &[1, 2, 5]);
    assert_eq!(writer.uncommitted_mut(), &[6]);

    drop(writer);
    assert_eq!(reader.as_slice(), &[1, 2, 5]);
}

#[test]
#[should_panic(expected = "Cannot commit more elements than have been initialized")]
fn test_commit_partial_panic() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    assert_eq!(writer.try_push(1), Ok(()));
    writer.commit_partial(2);
}

#[test]
fn test_extend() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader().clone();

    assert!(writer.extend([1, 2, 3]).is_ok());
    assert_eq!(writer.as_slice(), &[1, 2, 3]);
    writer.commit();
    assert_eq!(reader.as_slice(), &[1, 2, 3]);

    let mut remainder = writer.extend([4, 5]).unwrap_err();
    assert_eq!(writer.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(remainder.next(), Some(5));
}

#[test]
fn test_drop() {
    struct DropCounter<'a>(&'a AtomicUsize);

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drop_count = &AtomicUsize::new(0);

    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader().clone();

    assert!(writer.try_push(DropCounter(drop_count)).is_ok());
    assert!(writer.try_push(DropCounter(drop_count)).is_ok());
    writer.commit();

    assert!(writer.try_push(DropCounter(drop_count)).is_ok());
    writer.revert();
    assert_eq!(drop_count.load(Ordering::Relaxed), 1);

    // this one won't be committed, so should be dropped when the writer is dropped
    assert!(writer.try_push(DropCounter(drop_count)).is_ok());
    drop(writer);
    assert_eq!(drop_count.load(Ordering::Relaxed), 2);

    drop(reader);
    assert_eq!(drop_count.load(Ordering::Relaxed), 4);
}

#[test]
fn test_concurrent_read() {
    extern crate std;
    use std::thread;

    let mut writer = OnceArrayWriter::<usize>::with_capacity(1024);
    let reader = writer.reader().clone();

    let handle = thread::spawn(move || {
        while reader.len() < 1024 {
            let slice = reader.as_slice();
            // every committed element should equal its index
            for (i, &v) in slice.iter().enumerate() {
                assert_eq!(v, i);
            }
        }
    });

    for i in 0..1024 {
        writer.try_push(i).unwrap();
        writer.commit();
    }
    handle.join().unwrap();
}
