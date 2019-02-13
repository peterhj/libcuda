#[cfg(feature = "cuda_sys")]
extern crate cuda_sys;
#[cfg(not(feature = "cuda_sys"))]
#[macro_use] extern crate static_assertions;

pub mod blas;
pub mod driver;
pub mod extras;
#[cfg(not(feature = "cuda_sys"))]
pub mod ffi;
#[cfg(feature = "cuda_sys")]
pub mod ffi {
  pub use crate::ffi_via_cuda_sys::*;
}
#[cfg(feature = "cuda_sys")]
mod ffi_via_cuda_sys;
#[cfg(not(feature = "cuda_sys"))]
pub mod rand;
pub mod runtime;
