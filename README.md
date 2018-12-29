These are Rust bindings to the CUDA driver API.

The FFI bindings are done via [bindgen](https://github.com/rust-lang/rust-bindgen)
and are substantially whitelisted; see `build.rs` for the whitelisted APIs.
A small number of high-level wrappers are exported at the crate top-level.

There used to also be runtime API bindings in this crate, but those have been
moved to a separate [cudart](https://github.com/peterhj/cudart) crate.
