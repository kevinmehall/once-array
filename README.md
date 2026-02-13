# OnceArray

[Documentation](https://docs.rs/once-array) | [Release Notes](https://github.com/kevinmehall/once-array/releases)

A single-producer multiple-consumer append-only fixed capacity array in Rust.

Creating a `OnceArrayWriter<T>` allocates a fixed-capacity buffer and represents exclusive access to append elements. Any number of `Arc<OnceArray<T>>` references can be created and shared across threads. These readers can access the slice of committed elements, and see new elements as they are committed by the writer without any locking.

`OnceArray` serves as a building block for streaming data to multiple consumers while amortizing the cost of allocation and synchronization across chunks of many elements.

## License

[MIT](./LICENSE-MIT) or [Apache 2.0](./LICENSE-APACHE) at your option
