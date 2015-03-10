#![allow(non_camel_case_types)]

use libc::{c_float, c_double};

#[derive(Copy)]
#[repr(C)]
pub struct float2 {
  float0: c_float,
  float1: c_float,
}

#[derive(Copy)]
#[repr(C)]
pub struct double2 {
  float0: c_double,
  float1: c_double,
}

pub type cuComplex = float2;
pub type cuDoubleComplex = double2;
