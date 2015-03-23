use ffi::blas::*;

use libc::{c_int};

pub struct CublasError {
  status_code: cublasStatus_t,
}

impl CublasError {
  pub fn new<T>(x: T, status_code: cublasStatus_t) -> Result<T, CublasError> {
    match status_code {
      cublasStatus_t::Success => Ok(x),
      c => Err(CublasError{status_code: c}),
    }
  }
}

pub struct CublasHandle {
  ptr: cublasHandle_t,
}

impl CublasHandle {
  pub fn create() -> Result<CublasHandle, CublasError> {
    let mut handle: cublasHandle_t = 0 as cublasHandle_t;
    let status_code = unsafe { cublasCreate_v2(&mut handle as *mut cublasHandle_t) };
    CublasError::new(CublasHandle{ptr: handle}, status_code)
  }
}

pub fn cublas_sgemv(
  handle: &CublasHandle,
  a_trans: bool,
  m: usize, n: usize,
  alpha: f32,
  a: *const f32, lda: usize,
  x: *const f32, incx: usize,
  beta: f32,
  y: *mut f32, incy: usize,
) -> Result<(), CublasError>
{
  let op_a = match a_trans {
    false => cublasOperation_t::N,
    true => cublasOperation_t::T,
  };
  let status_code = unsafe {
    cublasSgemv_v2(
      handle.ptr,
      op_a,
      m as c_int, n as c_int,
      &alpha as *const f32,
      a, lda as c_int,
      x, incx as c_int,
      &beta as *const f32,
      y, incy as c_int,
    )
  };
  CublasError::new((), status_code)
}

pub fn cublas_sgemm(
  handle: &CublasHandle,
  a_trans: bool, b_trans: bool,
  m: usize, n: usize, k: usize,
  alpha: f32,
  a: *const f32, lda: usize,
  b: *const f32, ldb: usize,
  beta: f32,
  c: *mut f32, ldc: usize,
) -> Result<(), CublasError>
{
  let op_a = match a_trans {
    false => cublasOperation_t::N,
    true => cublasOperation_t::T,
  };
  let op_b = match b_trans {
    false => cublasOperation_t::N,
    true => cublasOperation_t::T,
  };
  let status_code = unsafe {
    cublasSgemm_v2(
      handle.ptr,
      op_a, op_b,
      m as c_int, n as c_int, k as c_int,
      &alpha as *const f32,
      a, lda as c_int,
      b, ldb as c_int,
      &beta as *const f32,
      c, ldc as c_int,
    )
  };
  CublasError::new((), status_code)
}
