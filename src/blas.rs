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
  Host,
  Device,
}

impl CublasPointerMode {
  pub fn to_ffi(&self) -> cublasPointerMode_t {
    match self {
      &CublasPointerMode::Host   => cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
      &CublasPointerMode::Device => cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
    }
  }
}

pub enum CublasTranspose {
  N,
  T,
  H,
}

impl CublasTranspose {
  pub fn to_ffi(&self) -> cublasOperation_t {
    match self {
      &CublasTranspose::N => cublasOperation_t::CUBLAS_OP_N,
      &CublasTranspose::T => cublasOperation_t::CUBLAS_OP_T,
      &CublasTranspose::H => cublasOperation_t::CUBLAS_OP_C,
    }
  }
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
    let status_code = unsafe { cublasSetPointerMode_v2(self.ptr, pointer_mode.to_ffi()) };
    match status_code {
      cublasStatus_t::Success => Ok(()),
      c => Err(CublasError::new(c)),
    }
  }
}

pub unsafe fn cublas_saxpy(
  handle: &CublasHandle,
  n: usize,
  alpha: f32,
  x: *const f32, incx: usize,
  y: *mut f32, incy: usize,
) -> CublasResult<()>
{
  let status_code = {
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

pub unsafe fn cublas_scopy(
  handle: &CublasHandle,
  n: usize,
  x: *const f32, incx: usize,
  y: *mut f32, incy: usize,
) -> CublasResult<()>
{
  let status_code = {
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

pub unsafe fn cublas_sscal(
  handle: &CublasHandle,
  n: usize,
  alpha: f32,
  x: *mut f32, incx: usize,
) -> CublasResult<()>
{
  let status_code = {
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

pub unsafe fn cublas_sgemv(
  handle: &CublasHandle,
  a_trans: CublasTranspose,
  m: usize, n: usize,
  alpha: f32,
  a: *const f32, lda: usize,
  x: *const f32, incx: usize,
  beta: f32,
  y: *mut f32, incy: usize,
) -> CublasResult<()>
{
  let status_code = {
    // FIXME: scalar passing convention depends on current context settings.
    cublasSgemv_v2(
      handle.ptr,
      a_trans.to_ffi(),
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

pub unsafe fn cublas_sgemm(
  handle: &CublasHandle,
  a_trans: CublasTranspose, b_trans: CublasTranspose,
  m: usize, n: usize, k: usize,
  alpha: f32,
  a: *const f32, lda: usize,
  b: *const f32, ldb: usize,
  beta: f32,
  c: *mut f32, ldc: usize,
) -> CublasResult<()>
{
  let status_code = {
    // FIXME: scalar passing convention depends on current context settings.
    cublasSgemm_v2(
      handle.ptr,
      a_trans.to_ffi(), b_trans.to_ffi(),
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
