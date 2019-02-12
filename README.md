These are Rust bindings to the CUDA toolkit APIs.

[Documentation (master, CUDA 10.0)](https://peterhj.github.io/libcuda-docs/cuda/)

The FFI bindings are done via [bindgen](https://github.com/rust-lang/rust-bindgen)
and are substantially whitelisted; see `build.rs` for the whitelisted APIs.
High-level wrappers are located in top-level modules (`driver`, `runtime`,
`blas`, and `rand`).
