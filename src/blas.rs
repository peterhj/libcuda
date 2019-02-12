use crate::ffi::cublas::*;
use crate::ffi::driver_types::{cudaStream_t};
use crate::runtime::{CudaStream};

use std::fmt;
use std::ptr::{null_mut};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CublasError(pub cublasStatus_t);

impl fmt::Debug for CublasError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    let code = self.get_code();
    match self.get_name() {
      Some(name) => write!(f, "CublasError({}, name={})", code, name),
      None => write!(f, "CublasError({})", code),
    }
  }
}

impl CublasError {
  pub fn get_code(&self) -> u32 {
    self.0
  }

  pub fn get_name(&self) -> Option<&'static str> {
    match self.0 {
      CUBLAS_STATUS_SUCCESS => Some("CUBLAS_STATUS_SUCCESS"),
      CUBLAS_STATUS_NOT_INITIALIZED => Some("CUBLAS_STATUS_NOT_INITIALIZED"),
      CUBLAS_STATUS_ALLOC_FAILED => Some("CUBLAS_STATUS_ALLOC_FAILED"),
      CUBLAS_STATUS_INVALID_VALUE => Some("CUBLAS_STATUS_INVALID_VALUE"),
      CUBLAS_STATUS_ARCH_MISMATCH => Some("CUBLAS_STATUS_ARCH_MISMATCH"),
      CUBLAS_STATUS_MAPPING_ERROR => Some("CUBLAS_STATUS_MAPPING_ERROR"),
      CUBLAS_STATUS_EXECUTION_FAILED => Some("CUBLAS_STATUS_EXECUTION_FAILED"),
      CUBLAS_STATUS_INTERNAL_ERROR => Some("CUBLAS_STATUS_INTERNAL_ERROR"),
      CUBLAS_STATUS_NOT_SUPPORTED => Some("CUBLAS_STATUS_NOT_SUPPORTED"),
      CUBLAS_STATUS_LICENSE_ERROR => Some("CUBLAS_STATUS_LICENSE_ERROR"),
      _ => None,
    }
  }
}

pub type CublasResult<T=()> = Result<T, CublasError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CublasPointerMode {
  Host,
  Device,
}

impl CublasPointerMode {
  pub fn to_raw(&self) -> cublasPointerMode_t {
    match self {
      &CublasPointerMode::Host => CUBLAS_POINTER_MODE_HOST,
      &CublasPointerMode::Device => CUBLAS_POINTER_MODE_DEVICE,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CublasAtomicsMode {
  NotAllowed,
  Allowed,
}

impl CublasAtomicsMode {
  pub fn to_raw(&self) -> cublasAtomicsMode_t {
    match self {
      &CublasAtomicsMode::NotAllowed => CUBLAS_ATOMICS_NOT_ALLOWED,
      &CublasAtomicsMode::Allowed => CUBLAS_ATOMICS_ALLOWED,
    }
  }
}

#[cfg(feature = "cuda_gte_9_0")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CublasMathMode {
  Default,
  TensorOp,
}

#[cfg(feature = "cuda_gte_9_0")]
impl CublasMathMode {
  pub fn to_raw(&self) -> cublasMath_t {
    match self {
      &CublasMathMode::Default => CUBLAS_DEFAULT_MATH,
      &CublasMathMode::TensorOp => CUBLAS_TENSOR_OP_MATH,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CublasTranspose {
  N,
  T,
  C,
}

impl CublasTranspose {
  pub fn to_raw(&self) -> cublasOperation_t {
    match self {
      &CublasTranspose::N => CUBLAS_OP_N,
      &CublasTranspose::T => CUBLAS_OP_T,
      &CublasTranspose::C => CUBLAS_OP_C,
    }
  }
}

pub struct CublasHandle {
  ptr:  cublasHandle_t,
}

impl Drop for CublasHandle {
  fn drop(&mut self) {
    assert!(!self.ptr.is_null());
    let status = unsafe { cublasDestroy_v2(self.ptr) };
    match status {
      CUBLAS_STATUS_SUCCESS => {}
      _ => panic!("cublasDestroy: {:?}", CublasError(status)),
    }
  }
}

impl CublasHandle {
  pub fn create() -> CublasResult<CublasHandle> {
    let mut stream: cublasHandle_t = null_mut();
    let status = unsafe { cublasCreate_v2(&mut stream as *mut _) };
    match status {
      CUBLAS_STATUS_SUCCESS => {}
      _ => return Err(CublasError(status)),
    }
    Ok(CublasHandle{
      ptr:  stream,
    })
  }

  pub fn get_version(&mut self) -> CublasResult<i32> {
    let mut version: i32 = 0;
    let status = unsafe { cublasGetVersion_v2(
        self.ptr,
        &mut version as *mut _,
    ) };
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(version),
      _ => Err(CublasError(status)),
    }
  }

  pub unsafe fn set_cuda_stream_raw(&mut self, stream: cudaStream_t) -> CublasResult {
    let status = cublasSetStream_v2(
        self.ptr,
        stream,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  pub fn set_cuda_stream(&mut self, stream: &mut CudaStream) -> CublasResult {
    let status = unsafe { cublasSetStream_v2(
        self.ptr,
        stream.as_raw(),
    ) };
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  pub fn set_pointer_mode(&mut self, mode: CublasPointerMode) -> CublasResult {
    let status = unsafe { cublasSetPointerMode_v2(
        self.ptr,
        mode.to_raw(),
    ) };
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  pub fn set_atomics_mode(&mut self, mode: CublasAtomicsMode) -> CublasResult {
    let status = unsafe { cublasSetAtomicsMode(
        self.ptr,
        mode.to_raw(),
    ) };
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  #[cfg(feature = "cuda_gte_9_0")]
  pub fn set_math_mode(&mut self, mode: CublasMathMode) -> CublasResult {
    let status = unsafe { cublasSetMathMode(
        self.ptr,
        mode.to_raw(),
    ) };
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }
}

pub trait CublasLevel1<T> {
  unsafe fn axpy(&mut self,
      num: i32,
      alpha: *const T,
      x: *const T,
      incx: i32,
      y: *mut T,
      incy: i32,
  ) -> CublasResult;
  unsafe fn dot(&mut self,
      num: i32,
      x: *const T,
      incx: i32,
      y: *const T,
      incy: i32,
      result: *mut T,
  ) -> CublasResult;
  unsafe fn nrm2(&mut self,
      num: i32,
      x: *const T,
      incx: i32,
      result: *mut T,
  ) -> CublasResult;
  unsafe fn scal(&mut self,
      num: i32,
      alpha: *const T,
      x: *mut T,
      incx: i32,
  ) -> CublasResult;
}

impl CublasLevel1<f32> for CublasHandle {
  unsafe fn axpy(&mut self,
      num: i32,
      alpha: *const f32,
      x: *const f32,
      incx: i32,
      y: *mut f32,
      incy: i32,
  ) -> CublasResult {
    let status = cublasSaxpy_v2(
        self.ptr,
        num,
        alpha,
        x,
        incx,
        y,
        incy,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  unsafe fn dot(&mut self,
      num: i32,
      x: *const f32,
      incx: i32,
      y: *const f32,
      incy: i32,
      result: *mut f32,
  ) -> CublasResult {
    let status = cublasSdot_v2(
        self.ptr,
        num,
        x,
        incx,
        y,
        incy,
        result,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  unsafe fn nrm2(&mut self,
      num: i32,
      x: *const f32,
      incx: i32,
      result: *mut f32,
  ) -> CublasResult {
    let status = cublasSnrm2_v2(
        self.ptr,
        num,
        x,
        incx,
        result,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }

  unsafe fn scal(&mut self,
      num: i32,
      alpha: *const f32,
      x: *mut f32,
      incx: i32,
  ) -> CublasResult {
    let status = cublasSscal_v2(
        self.ptr,
        num,
        alpha,
        x,
        incx,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }
}

pub trait CublasLevel2<T> {
  unsafe fn gemv(&mut self,
      transpose: CublasTranspose,
      rows: i32,
      cols: i32,
      alpha: *const T,
      a: *const T,
      lda: i32,
      x: *const T,
      incx: i32,
      beta: *const T,
      y: *mut T,
      incy: i32,
  ) -> CublasResult;
}

impl CublasLevel2<f32> for CublasHandle {
  unsafe fn gemv(&mut self,
      transpose: CublasTranspose,
      rows: i32,
      cols: i32,
      alpha: *const f32,
      a: *const f32,
      lda: i32,
      x: *const f32,
      incx: i32,
      beta: *const f32,
      y: *mut f32,
      incy: i32,
  ) -> CublasResult {
    let status = cublasSgemv_v2(
        self.ptr,
        transpose.to_raw(),
        rows,
        cols,
        alpha,
        a,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }
}

pub trait CublasLevel3<T> {
  unsafe fn gemm(&mut self,
      transpose_a: CublasTranspose,
      transpose_b: CublasTranspose,
      rows: i32,
      cols: i32,
      inner: i32,
      alpha: *const T,
      a: *const T,
      lda: i32,
      b: *const T,
      ldb: i32,
      beta: *const T,
      c: *mut T,
      ldc: i32,
  ) -> CublasResult;
}

impl CublasLevel3<f32> for CublasHandle {
  unsafe fn gemm(&mut self,
      transpose_a: CublasTranspose,
      transpose_b: CublasTranspose,
      rows: i32,
      cols: i32,
      inner_dim: i32,
      alpha: *const f32,
      a: *const f32,
      lda: i32,
      b: *const f32,
      ldb: i32,
      beta: *const f32,
      c: *mut f32,
      ldc: i32,
  ) -> CublasResult {
    let status = cublasSgemm_v2(
        self.ptr,
        transpose_a.to_raw(),
        transpose_b.to_raw(),
        rows,
        cols,
        inner_dim,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    );
    match status {
      CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(status)),
    }
  }
}
