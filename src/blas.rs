use ffi::blas::*;
use runtime::{CudaStream};

use libc::{c_int};

#[derive(Clone, Copy, Debug)]
pub struct CublasError {
  status_code: cublasStatus_t,
}

impl CublasError {
  pub fn new(status_code: cublasStatus_t) -> CublasError {
    match status_code {
      cublasStatus_t::Success => unreachable!(),
      c => CublasError{status_code: c},
    }
  }
}

pub type CublasResult<T> = Result<T, CublasError>;

pub enum CublasPointerMode {
  HostPointers,
  DevicePointers,
}

pub struct CublasHandle {
  ptr: cublasHandle_t,
}

impl CublasHandle {
  pub fn create() -> CublasResult<CublasHandle> {
    let mut handle: cublasHandle_t = 0 as cublasHandle_t;
    let status_code = unsafe { cublasCreate_v2(&mut handle as *mut cublasHandle_t) };
    match status_code {
      cublasStatus_t::Success => Ok(CublasHandle{ptr: handle}),
      c => Err(CublasError::new(c)),
    }
  }

  pub fn set_stream(&self, stream: &CudaStream) -> CublasResult<()> {
    let status_code = unsafe { cublasSetStream_v2(self.ptr, stream.ptr) };
    match status_code {
      cublasStatus_t::Success => Ok(()),
      c => Err(CublasError::new(c)),
    }
  }

  pub fn set_pointer_mode(&self, pointer_mode: CublasPointerMode) -> CublasResult<()> {
    let mode = match pointer_mode {
      CublasPointerMode::HostPointers   => cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
      CublasPointerMode::DevicePointers => cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
    };
    let status_code = unsafe { cublasSetPointerMode_v2(self.ptr, mode) };
    match status_code {
      cublasStatus_t::Success => Ok(()),
      c => Err(CublasError::new(c)),
    }
  }
}

pub fn cublas_saxpy(
  handle: &CublasHandle,
  n: usize,
  alpha: f32,
  x: *const f32, incx: usize,
  y: *mut f32, incy: usize,
) -> CublasResult<()>
{
  let status_code = unsafe {
    cublasSaxpy_v2(
      handle.ptr,
      n as c_int,
      &alpha as *const f32,
      x, incx as c_int,
      y, incy as c_int,
    )
  };
  match status_code {
    cublasStatus_t::Success => Ok(()),
    c => Err(CublasError::new(c)),
  }
}

pub fn cublas_scopy(
  handle: &CublasHandle,
  n: usize,
  x: *const f32, incx: usize,
  y: *mut f32, incy: usize,
) -> CublasResult<()>
{
  let status_code = unsafe {
    cublasScopy_v2(
      handle.ptr,
      n as c_int,
      x, incx as c_int,
      y, incy as c_int,
    )
  };
  match status_code {
    cublasStatus_t::Success => Ok(()),
    c => Err(CublasError::new(c)),
  }
}

pub fn cublas_sscal(
  handle: &CublasHandle,
  n: usize,
  alpha: f32,
  x: *mut f32, incx: usize,
) -> CublasResult<()>
{
  let status_code = unsafe {
    // FIXME: scalar passing convention depends on current context settings.
    cublasSscal_v2(
      handle.ptr,
      n as c_int,
      &alpha as *const f32,
      x, incx as c_int,
    )
  };
  match status_code {
    cublasStatus_t::Success => Ok(()),
    c => Err(CublasError::new(c)),
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
) -> CublasResult<()>
{
  let op_a = match a_trans {
    false => cublasOperation_t::CUBLAS_OP_N,
    true => cublasOperation_t::CUBLAS_OP_T,
  };
  let status_code = unsafe {
    // FIXME: scalar passing convention depends on current context settings.
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
  match status_code {
    cublasStatus_t::Success => Ok(()),
    c => Err(CublasError::new(c)),
  }
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
) -> CublasResult<()>
{
  let op_a = match a_trans {
    false => cublasOperation_t::CUBLAS_OP_N,
    true => cublasOperation_t::CUBLAS_OP_T,
  };
  let op_b = match b_trans {
    false => cublasOperation_t::CUBLAS_OP_N,
    true => cublasOperation_t::CUBLAS_OP_T,
  };
  let status_code = unsafe {
    // FIXME: scalar passing convention depends on current context settings.
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
  match status_code {
    cublasStatus_t::Success => Ok(()),
    c => Err(CublasError::new(c)),
  }
}
