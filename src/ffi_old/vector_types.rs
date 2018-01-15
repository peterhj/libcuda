#![allow(non_camel_case_types)]

use std::os::raw::{c_float, c_double};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct float2 {
  float0: c_float,
  float1: c_float,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct double2 {
  float0: c_double,
  float1: c_double,
}

pub type cuComplex = float2;
pub type cuDoubleComplex = double2;
