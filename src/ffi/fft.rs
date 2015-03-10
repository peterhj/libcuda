#![allow(missing_copy_implementations)]
#![allow(non_camel_case_types)]

use ffi::runtime::{cudaStream_t};
use ffi::vector_types::{cuComplex, cuDoubleComplex};

use libc::{
  c_void, c_int, c_uint, c_float, c_double, size_t,
};

#[derive(Copy)]
#[repr(C)]
pub enum cufftResult {
  Success                 = 0,
  InvalidPlan             = 1,
  AllocFailed             = 2,
  InvalidType             = 3,
  InvalidValue            = 4,
  InternalError           = 5,
  ExecFailed              = 6,
  SetupFailed             = 7,
  InvalidSize             = 8,
  UnalignedData           = 9,
  IncompleteParameterList = 10,
  InvalidDevice           = 11,
  ParseError              = 12,
  NoWorkspace             = 13,
}

#[derive(Copy)]
#[repr(C)]
pub enum cufftType {
  R2C = 0x2a,
  C2R = 0x2c,
  C2C = 0x29,
  D2Z = 0x6a,
  Z2D = 0x6c,
  Z2Z = 0x69,
}

pub type cufftHandle = c_uint;
pub type cufftReal = c_float;
pub type cufftDoubleReal = c_double;
pub type cufftComplex = cuComplex;
pub type cufftDoubleComplex = cuDoubleComplex;

#[derive(Copy)]
#[repr(C)]
pub enum cufftCompatibility {
  Native          = 0,
  FFTWPadding     = 1,
  FFTWAsymmetric  = 2,
  FFTWAll         = 3,
}

#[link(name = "cufft")]
extern "C" {
  // Basic Plans
  pub fn cufftPlan1d(plan: *mut cufftHandle, nx: c_int, ty: cufftType, batch: c_int) -> cufftResult;
  pub fn cufftPlan2d(plan: *mut cufftHandle, nx: c_int, ny: c_int, ty: cufftType) -> cufftResult;
  pub fn cufftPlan3d(plan: *mut cufftHandle, nx: c_int, ny: c_int, nz: c_int, ty: cufftType) -> cufftResult;
  pub fn cufftPlanMany(plan: *mut cufftHandle, rank: c_int, n: *mut c_int, inembed: *mut c_int, istride: c_int, idist: c_int, onembed: *mut c_int, ostride: c_int, odist: c_int, ty: cufftType, batch: c_int) -> cufftResult;

  // Extensible Plans
  pub fn cufftCreate(plan: *mut cufftHandle) -> cufftResult;
  pub fn cufftMakePlan1d(plan: cufftHandle, nx: c_int, ty: cufftType, batch: c_int, workSize: *mut size_t) -> cufftResult;
  pub fn cufftMakePlan2d(plan: cufftHandle, nx: c_int, ny: c_int, ty: cufftType, workSize: *mut size_t) -> cufftResult;
  pub fn cufftMakePlan3d(plan: cufftHandle, nx: c_int, ny: c_int, nz: c_int, ty: cufftType, workSize: *mut size_t) -> cufftResult;
  pub fn cufftMakePlanMany(plan: cufftHandle, rank: c_int, n: *mut c_int, inembed: *mut c_int, istride: c_int, idist: c_int, onembed: *mut c_int, ostride: c_int, odist: c_int, ty: cufftType, batch: c_int, workSize: *mut size_t) -> cufftResult;

  // Estimated Size of Work Area
  pub fn cufftEstimate1d(nx: c_int, ty: cufftType, batch: c_int, workSize: *mut size_t) -> cufftResult;
  pub fn cufftEstimate2d(nx: c_int, ny: c_int, ty: cufftType, workSize: *mut size_t) -> cufftResult;
  // TODO

  // Refined Estimated Size of Work Area
  pub fn cufftGetSize1d(plan: cufftHandle, nx: c_int, ty: cufftType, batch: c_int, workSize: *mut size_t) -> cufftResult;
  pub fn cufftGetSize2d(plan: cufftHandle, nx: c_int, ny: c_int, ty: cufftType, workSize: *mut size_t) -> cufftResult;
  // TODO

  // Actual Size of Work Area
  pub fn cufftGetSize(plan: cufftHandle, workSize: *mut size_t) -> cufftResult;

  // Caller Allocated Work Area Support
  pub fn cufftSetAutoAllocation(plan: cufftHandle, autoAllocate: c_int) -> cufftResult;
  pub fn cufftSetWorkArea(plan: cufftHandle, workArea: *mut c_void) -> cufftResult;

  // Execution
  pub fn cufftExecC2C(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut cufftComplex, direction: c_int) -> cufftResult;
  pub fn cufftExecZ2Z(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut cufftDoubleComplex, direction: c_int) -> cufftResult;
  pub fn cufftExecR2C(plan: cufftHandle, idata: *mut cufftReal, odata: *mut cufftComplex, direction: c_int) -> cufftResult;
  pub fn cufftExecD2Z(plan: cufftHandle, idata: *mut cufftDoubleReal, odata: *mut cufftDoubleComplex, direction: c_int) -> cufftResult;

  // Multiple GPUs
  // TODO

  // Callbacks
  // TODO

  // Stream
  pub fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult;

  // Version
  pub fn cufftGetVersion(version: *mut c_int) -> cufftResult;

  // Data Layout Compatibility
  pub fn cufftSetCompatibilityMode(plan: cufftHandle, mode: cufftCompatibility) -> cufftResult;
}
