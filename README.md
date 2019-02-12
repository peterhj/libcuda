These are Rust bindings to the CUDA toolkit APIs.

[https://peterhj.github.io/libcuda-docs/cuda/](Documentation)

The FFI bindings are done via [bindgen](https://github.com/rust-lang/rust-bindgen)
and are substantially whitelisted; see `build.rs` for the whitelisted APIs.
High-level wrappers are located in top-level modules (`driver`, `runtime`,
`blas`, and `rand`).
