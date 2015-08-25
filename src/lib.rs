#![feature(libc)]
#![feature(optin_builtin_traits)]

extern crate libc;

pub mod dnn;
pub mod driver;
pub mod ffi;
pub mod fft;
pub mod runtime;
#[cfg(test)]
mod tests;
