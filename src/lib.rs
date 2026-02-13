use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single-producer multiple-consumer append-only fixed capacity array.
pub struct OnceArray<T> {
    // safety invariants:
    // * data and cap may not change
    // * len may never decrease
    // * nothing may write to or invalidate *data..*data.add(len), because
    //   another thread may have a reference to it
    data: *mut T,
    cap: usize,
    len: AtomicUsize,
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

    /// Returns the maximum number of elements this buffer can hold.
    ///
    /// The capacity can't change once allocated.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns the current number of elements in the buffer.
    ///
    /// This increases when the [`OnceArrayWriter`] appends to the buffer,
    /// but can never decrease.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Obtain a slice of the written part of the buffer.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            // SAFETY: This came from a vector and is properly aligned.
            // The part up to len is initialized, and won't change
            std::slice::from_raw_parts(self.data, self.len())
        }
    }
}

impl<T> Deref for OnceArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> From<Vec<T>> for OnceArray<T> {
    fn from(val: Vec<T>) -> Self {
        OnceArray::from_vec(val)
    }
}

/// Exclusive write access to a [`OnceArray`].
pub struct OnceArrayWriter<T> {
    inner: Arc<OnceArray<T>>,
}

impl<T> OnceArrayWriter<T> {
    fn from_vec(v: Vec<T>) -> OnceArrayWriter<T> {
        Self {
            inner: Arc::new(OnceArray::from_vec(v)),
        }
    }

    /// Creates a new `OnceArrayWriter` with the specified capacity.
    pub fn with_capacity(n: usize) -> OnceArrayWriter<T> {
        Self::from_vec(Vec::with_capacity(n))
    }

    /// Obtain a read-only reference.
    pub fn reader(&self) -> Arc<OnceArray<T>> {
        self.inner.clone()
    }

    /// Returns the number of additional elements that can be written to the buffer before it is full.
    pub fn remaining_capacity(&self) -> usize {
        self.inner.cap - self.inner.len.load(Ordering::Relaxed)
    }

    /// Attempts to append an element to the buffer.
    ///
    /// If the buffer has capacity for an additional element, returns
    /// `Ok(index)` where `index` is the index of the newly appended element. If
    /// the buffer is full, returns `Err(val)`, returning ownership of the value
    /// that could not be added.
    pub fn try_push(&mut self, val: T) -> Result<usize, T> {
        let len = self.inner.len.load(Ordering::Relaxed);
        if len < self.inner.cap {
            unsafe {
                // SAFETY:
                // * checked that position is less than capacity so
                //   address is in bounds.
                // * this is above the current len so doesn't invalidate slices
                // * this has &mut exclusive access to the only `BufferChunkWriter`
                //   wrapping `inner`, so no other thread is writing.
                self.inner.data.add(len).write(val);
            }

            self.len.store(len + 1, Ordering::Release);
            Ok(len)
        } else {
            Err(val)
        }
    }

    /// Attempts to append elements from `src` to the array.
    ///
    /// Returns the tail of the slice that could not be written to the buffer.
    /// If the buffer is not full, and all elements were written, this will be
    /// an empty slice.
    ///
    /// The new elements become visible to readers atomically.
    pub fn extend_from_slice<'a>(&mut self, src: &'a [T]) -> &'a [T]
    where
        T: Copy,
    {
        let len = self.inner.len.load(Ordering::Relaxed);
        let count = self.inner.cap.saturating_sub(len).min(src.len());
        unsafe {
            // SAFETY:
            // * checked that position is less than capacity so
            //   address is in bounds.
            // * this is above the current len so doesn't invalidate slices
            // * this has &mut exclusive access to the only `BufferChunkWriter`
            //   wrapping `inner`, so no other thread is writing.
            self.inner
                .data
                .add(len)
                .copy_from_nonoverlapping(src.as_ptr(), count);
        }

        self.len.store(len + count, Ordering::Release);
        &src[count..]
    }
}

impl<T> Deref for OnceArrayWriter<T> {
    type Target = OnceArray<T>;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<T> From<Vec<T>> for OnceArrayWriter<T> {
    fn from(vec: Vec<T>) -> OnceArrayWriter<T> {
        OnceArrayWriter::from_vec(vec)
    }
}

#[test]
fn test_push() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader();
    assert_eq!(reader.capacity(), 4);
    assert_eq!(reader.len(), 0);

    assert_eq!(writer.try_push(1), Ok(0));
    assert_eq!(reader.len(), 1);
    assert_eq!(reader.as_slice(), &[1]);

    assert_eq!(writer.try_push(2), Ok(1));
    assert_eq!(writer.try_push(3), Ok(2));
    assert_eq!(writer.try_push(4), Ok(3));
    assert_eq!(writer.try_push(5), Err(5));

    assert_eq!(reader.len(), 4);
    assert_eq!(reader.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_extend_from_slice() {
    let mut writer = OnceArrayWriter::with_capacity(4);
    let reader = writer.reader();
    assert_eq!(reader.capacity(), 4);
    assert_eq!(reader.len(), 0);

    assert_eq!(writer.extend_from_slice(&[1, 2]), &[]);
    assert_eq!(reader.len(), 2);
    assert_eq!(reader.as_slice(), &[1, 2]);

    assert_eq!(writer.extend_from_slice(&[3, 4, 5, 6]), &[5, 6]);
    assert_eq!(reader.len(), 4);
    assert_eq!(reader.as_slice(), &[1, 2, 3, 4]);
}
