#![allow(non_camel_case_types)]

use ffi::runtime::{cudaStream_t};
use ffi::vector_types::{cuComplex};

use libc::{
  c_int, c_float,
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cublasStatus_t {
  Success         = 0,
  NotInitialized  = 1,
  AllocFailed     = 3,
  InvalidValue    = 7,
  ArchMismatch    = 8,
  MappingError    = 11,
  ExecutionFailed = 13,
  InternalError   = 14,
  NotSupported    = 15,
  LicenseError    = 16,
}

#[repr(C)]
struct cublasContext;
pub type cublasHandle_t = *mut cublasContext;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasPointerMode_t {
  CUBLAS_POINTER_MODE_HOST    = 0,
  CUBLAS_POINTER_MODE_DEVICE  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasAtomicsMode_t {
  CUBLAS_ATOMICS_NOT_ALLOWED  = 0,
  CUBLAS_ATOMICS_ALLOWED      = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasFillMode_t {
  CUBLAS_FILL_MODE_LOWER  = 0,
  CUBLAS_FILL_MODE_UPPER  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasDiagType_t {
  CUBLAS_DIAG_NON_UNIT  = 0,
  CUBLAS_DIAG_UNIT      = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasSideMode_t {
  CUBLAS_SIDE_LEFT  = 0,
  CUBLAS_SIDE_RIGHT = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cublasOperation_t {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2,
}

#[link(name = "cublas", kind = "dylib")]
extern "C" {
  // Helper Functions
  pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
  pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
  pub fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;
  pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
  pub fn cublasGetStream_v2(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t;
  pub fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t;
  pub fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut cublasPointerMode_t) -> cublasStatus_t;
  pub fn cublasSetAtomicsMode_v2(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t;
  pub fn cublasGetAtomicsMode_v2(handle: cublasHandle_t, mode: *mut cublasAtomicsMode_t) -> cublasStatus_t;
  // TODO

  // Level 1 Functions
  pub fn cublasSaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *const f32, incx: c_int,
    y: *mut f32, incy: c_int) -> cublasStatus_t;
  pub fn cublasScopy_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const f32, incx: c_int,
    y: *mut f32, incy: c_int,
  ) -> cublasStatus_t;
  pub fn cublasSscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const f32,
    x: *mut f32, incx: c_int,
  ) -> cublasStatus_t;
  // TODO

  // Level 2 Functions
  pub fn cublasSgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int, n: c_int,
    alpha: *const f32,
    a: *const f32, lda: c_int,
    x: *const f32, incx: c_int,
    beta: *const f32,
    y: *mut f32, incy: c_int,
  ) -> cublasStatus_t;
  /*pub fn cublasCgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int, n: c_int,
    alpha: *const cuComplex,
    a: *const cuComplex, lda: c_int,
    x: *const cuComplex, incx: c_int,
    beta: *const cuComplex,
    y: *mut cuComplex, incy: c_int,
  ) -> cublasStatus_t;*/
  pub fn cublasSger_v2(
    handle: cublasHandle_t,
    m: c_int, n: c_int,
    alpha: *const f32,
    x: *const f32, incx: c_int,
    y: *const f32, incy: c_int,
    a: *mut f32, lda: c_int,
  ) -> cublasStatus_t;
  pub fn cublasSsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    alpha: *const f32,
    x: *const f32, incx: c_int,
    a: *mut f32, lda: c_int,
  ) -> cublasStatus_t;

  // Level 3 Functions
  pub fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const f32,
    A: *const f32, lda: c_int,
    B: *const f32, ldb: c_int,
    beta: *const f32,
    C: *mut f32, ldc: c_int,
  ) -> cublasStatus_t;
  pub fn cublasSgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const f32,
    A: *mut *const f32, lda: c_int,
    B: *mut *const f32, ldb: c_int,
    beta: *const f32,
    C: *mut *mut f32, ldc: c_int,
    batchCount: c_int,
  ) -> cublasStatus_t;
  /*pub fn cublasCgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const cuComplex,
    A: *const cuComplex, lda: c_int,
    B: *const cuComplex, ldb: c_int,
    beta: *const cuComplex,
    C: *mut cuComplex, ldc: c_int,
  ) -> cublasStatus_t;
  pub fn cublasCgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t, transb: cublasOperation_t,
    m: c_int, n: c_int, k: c_int,
    alpha: *const cuComplex,
    A: *mut *const cuComplex, lda: c_int,
    B: *mut *const cuComplex, ldb: c_int,
    beta: *const cuComplex,
    C: *mut *mut cuComplex, ldc: c_int,
    batchCount: c_int,
  ) -> cublasStatus_t;*/
  pub fn cublasSsyrk(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int, k: c_int,
    alpha: *const f32,
    a: *const f32, lda: c_int,
    beta: *const f32,
    c: *mut f32, ldc: c_int,
  ) -> cublasStatus_t;
  // TODO
}
