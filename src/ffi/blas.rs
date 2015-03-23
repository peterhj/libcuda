#![allow(non_camel_case_types)]

use ffi::runtime::{cudaStream_t};
use ffi::vector_types::{cuComplex};

use libc::{
  c_int, c_float,
};

#[repr(C)]
struct cublasHandle;
pub type cublasHandle_t = *mut cublasHandle;

#[derive(Copy)]
#[repr(C)]
pub enum cublasStatus_t {
  Success         = 0,
  NotInitialized  = 1,
  AllocFailed     = 2,
  InvalidValue    = 3,
  ArchMismatch    = 4,
  MappingError    = 5,
  ExecutionFailed = 6,
  InternalError   = 7,
  NotSupported    = 8,
  LicenseError    = 9,
}

#[derive(Copy)]
#[repr(C)]
pub enum cublasOperation_t {
  N = 0,
  T = 1,
  C = 2,
}

#[link(name = "cublas", kind = "dylib")]
extern "C" {
  // Helper Functions
  pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
  pub fn cublasDestroy(handle: cublasHandle_t) -> cublasStatus_t;
  pub fn cublasGetVersion(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;
  pub fn cublasSetStream(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
  pub fn cublasGetStream(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t;
  // TODO

  // Level 1 Functions
  // TODO

  // Level 2 Functions
  pub fn cublasSgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int, n: c_int,
    alpha: *const c_float,
    a: *const c_float, lda: c_int,
    x: *const c_float, incx: c_int,
    beta: *const c_float,
    y: *mut c_float, incy: c_int) -> cublasStatus_t;
  pub fn cublasCgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int, n: c_int,
    alpha: *const cuComplex,
    a: *const cuComplex, lda: c_int,
    x: *const cuComplex, incx: c_int,
    beta: *const cuComplex,
    y: *mut cuComplex, incy: c_int) -> cublasStatus_t;
  // TODO

  // Level 3 Functions
  pub fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const c_float,
    A: *const c_float, lda: c_int,
    B: *const c_float, ldb: c_int,
    beta: *const c_float,
    C: *mut c_float, ldc: c_int) -> cublasStatus_t;
  pub fn cublasSgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const c_float,
    A: *mut *const c_float, lda: c_int,
    B: *mut *const c_float, ldb: c_int,
    beta: *const c_float,
    C: *mut *mut c_float, ldc: c_int,
    batchCount: c_int) -> cublasStatus_t;
  pub fn cublasCgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const cuComplex,
    A: *const cuComplex, lda: c_int,
    B: *const cuComplex, ldb: c_int,
    beta: *const cuComplex,
    C: *mut cuComplex, ldc: c_int) -> cublasStatus_t;
  pub fn cublasCgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const cuComplex,
    A: *mut *const cuComplex, lda: c_int,
    B: *mut *const cuComplex, ldb: c_int,
    beta: *const cuComplex,
    C: *mut *mut cuComplex, ldc: c_int,
    batchCount: c_int) -> cublasStatus_t;
  // TODO
}
