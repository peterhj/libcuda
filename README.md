These are Rust bindings to CUDA.

The FFI bindings are done via [bindgen](https://github.com/rust-lang-nursery/rust-bindgen).
The Rust equivalents are defined in the respective module, e.g. `runtime` for
the CUDA runtime API. Bindings are substantially whitelisted; for example,
currently only the runtime API is exposed, and of that only a small set of
"core" runtime functionality.
