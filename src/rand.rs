use crate::ffi::curand::*;
use crate::ffi::driver_types::{cudaStream_t};
use crate::runtime::{CudaStream};

use std::fmt;
use std::marker::{PhantomData};
use std::ptr::{null_mut};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CurandError(pub curandStatus_t);

impl fmt::Debug for CurandError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    let code = self.get_code();
    match self.get_name() {
      Some(name) => write!(f, "CurandError({}, name={})", code, name),
      None => write!(f, "CurandError({})", code),
    }
  }
}

impl CurandError {
  pub fn get_code(&self) -> u32 {
    self.0
  }

  pub fn get_name(&self) -> Option<&'static str> {
    match self.0 {
      CURAND_STATUS_SUCCESS => Some("CURAND_STATUS_SUCCESS"),
      CURAND_STATUS_VERSION_MISMATCH => Some("CURAND_STATUS_VERSION_MISMATCH"),
      CURAND_STATUS_NOT_INITIALIZED => Some("CURAND_STATUS_NOT_INITIALIZED"),
      CURAND_STATUS_ALLOCATION_FAILED => Some("CURAND_STATUS_ALLOCATION_FAILED"),
      CURAND_STATUS_TYPE_ERROR => Some("CURAND_STATUS_TYPE_ERROR"),
      CURAND_STATUS_OUT_OF_RANGE => Some("CURAND_STATUS_OUT_OF_RANGE"),
      CURAND_STATUS_LENGTH_NOT_MULTIPLE => Some("CURAND_STATUS_LENGTH_NOT_MULTIPLE"),
      CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => Some("CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"),
      CURAND_STATUS_LAUNCH_FAILURE => Some("CURAND_STATUS_LAUNCH_FAILURE"),
      CURAND_STATUS_PREEXISTING_FAILURE => Some("CURAND_STATUS_PREEXISTING_FAILURE"),
      CURAND_STATUS_INITIALIZATION_FAILED => Some("CURAND_STATUS_INITIALIZATION_FAILED"),
      CURAND_STATUS_ARCH_MISMATCH => Some("CURAND_STATUS_ARCH_MISMATCH"),
      CURAND_STATUS_INTERNAL_ERROR => Some("CURAND_STATUS_INTERNAL_ERROR"),
      _ => None,
    }
  }
}

pub type CurandResult<T=()> = Result<T, CurandError>;

pub fn get_version() -> CurandResult<i32> {
  let mut version: i32 = 0;
  let status = unsafe { curandGetVersion(
      &mut version as *mut _,
  ) };
  match status {
    CURAND_STATUS_SUCCESS => Ok(version),
    _ => Err(CurandError(status)),
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurandRngOrdering {
  Default,
  Best,
  Seeded,
}

impl Default for CurandRngOrdering {
  fn default() -> CurandRngOrdering {
    CurandRngOrdering::Default
  }
}

impl CurandRngOrdering {
  pub fn to_raw(&self) -> curandOrdering_t {
    match self {
      &CurandRngOrdering::Default => CURAND_ORDERING_PSEUDO_DEFAULT,
      &CurandRngOrdering::Best    => CURAND_ORDERING_PSEUDO_BEST,
      &CurandRngOrdering::Seeded  => CURAND_ORDERING_PSEUDO_SEEDED,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurandQrngOrdering {
  Default,
}

impl Default for CurandQrngOrdering {
  fn default() -> CurandQrngOrdering {
    CurandQrngOrdering::Default
  }
}

impl CurandQrngOrdering {
  pub fn to_raw(&self) -> curandOrdering_t {
    match self {
      &CurandQrngOrdering::Default => CURAND_ORDERING_QUASI_DEFAULT,
    }
  }
}

pub enum CurandTestRngType {}
pub enum CurandDefaultRngType {}
pub enum CurandXorwowRngType {}
pub enum CurandMrg32k3aRngType {}
pub enum CurandMtgp32RngType {}
pub enum CurandMt19937RngType {}
pub enum CurandPhilox4x32_10RngType {}
pub enum CurandDefaultQrngType {}
pub enum CurandSobol32QrngType {}
pub enum CurandScrambledSobol32QrngType {}
pub enum CurandSobol64QrngType {}
pub enum CurandScrambledSobol64QrngType {}

pub trait CurandRngType {
  fn rng_type_raw() -> curandRngType_t;
}

pub trait CurandPseudoRngType: CurandRngType {}
pub trait CurandQuasiRngType: CurandRngType {}

impl CurandRngType for CurandTestRngType {
  fn rng_type_raw() -> curandRngType_t {
    CURAND_RNG_TEST
  }
}

impl CurandRngType for CurandDefaultRngType {
  fn rng_type_raw() -> curandRngType_t {
    CURAND_RNG_PSEUDO_DEFAULT
  }
}

impl CurandPseudoRngType for CurandDefaultRngType {}

impl CurandRngType for CurandMt19937RngType {
  fn rng_type_raw() -> curandRngType_t {
    CURAND_RNG_PSEUDO_MT19937
  }
}

impl CurandPseudoRngType for CurandMt19937RngType {}

impl CurandRngType for CurandPhilox4x32_10RngType {
  fn rng_type_raw() -> curandRngType_t {
    CURAND_RNG_PSEUDO_PHILOX4_32_10
  }
}

impl CurandPseudoRngType for CurandPhilox4x32_10RngType {}

impl CurandRngType for CurandDefaultQrngType {
  fn rng_type_raw() -> curandRngType_t {
    CURAND_RNG_QUASI_DEFAULT
  }
}

impl CurandQuasiRngType for CurandDefaultQrngType {}

pub type CurandTestRng = CurandGenerator<CurandTestRngType>;
pub type CurandDefaultRng = CurandGenerator<CurandDefaultRngType>;
pub type CurandMt19937Rng = CurandGenerator<CurandMt19937RngType>;
pub type CurandPhilox4x32_10Rng = CurandGenerator<CurandPhilox4x32_10RngType>;
pub type CurandDefaultQrng = CurandGenerator<CurandDefaultQrngType>;

pub struct CurandGenerator<RngType> {
  ptr:  curandGenerator_t,
  _mrk: PhantomData<fn (RngType)>,
}

impl<RngType> Drop for CurandGenerator<RngType> {
  fn drop(&mut self) {
    assert!(!self.ptr.is_null());
    let status = unsafe { curandDestroyGenerator(self.ptr) };
    match status {
      CURAND_STATUS_SUCCESS => {}
      _ => panic!("curandDestroyGenerator: {:?}", CurandError(status)),
    }
  }
}

impl<RngType: CurandPseudoRngType> CurandGenerator<RngType> {
  pub fn set_seed_from_u64(&mut self, seed: u64) -> CurandResult {
    let status = unsafe { curandSetPseudoRandomGeneratorSeed(
        self.ptr,
        seed,
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub fn set_pseudorandom_ordering(&mut self, ordering: CurandRngOrdering) -> CurandResult {
    let status = unsafe { curandSetGeneratorOrdering(
        self.ptr,
        ordering.to_raw(),
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }
}

impl<RngType: CurandQuasiRngType> CurandGenerator<RngType> {
  pub fn set_quasirandom_num_dims(&mut self, num_dims: u32) -> CurandResult {
    let status = unsafe { curandSetQuasiRandomGeneratorDimensions(
        self.ptr,
        num_dims,
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub fn set_quasirandom_ordering(&mut self, ordering: CurandQrngOrdering) -> CurandResult {
    let status = unsafe { curandSetGeneratorOrdering(
        self.ptr,
        ordering.to_raw(),
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }
}

impl<RngType: CurandRngType> CurandGenerator<RngType> {
  pub fn create() -> CurandResult<CurandGenerator<RngType>> {
    let mut generator: curandGenerator_t = null_mut();
    let status = unsafe { curandCreateGenerator(
        &mut generator as *mut _,
        RngType::rng_type_raw(),
    ) };
    match status {
      CURAND_STATUS_SUCCESS => {}
      _ => return Err(CurandError(status)),
    }
    Ok(CurandGenerator{
      ptr:  generator,
      _mrk: PhantomData,
    })
  }

  pub fn as_raw(&self) -> curandGenerator_t {
    self.ptr
  }

  pub fn ptr_eq(&self, other: &CurandGenerator<RngType>) -> bool {
    self.ptr == other.ptr
  }

  pub unsafe fn set_cuda_stream_raw(&mut self, stream: cudaStream_t) -> CurandResult {
    let status = curandSetStream(
        self.ptr,
        stream,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub fn set_cuda_stream(&mut self, stream: &mut CudaStream) -> CurandResult {
    let status = unsafe { curandSetStream(
        self.ptr,
        stream.as_raw(),
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub fn set_offset(&mut self, offset: u64) -> CurandResult {
    let status = unsafe { curandSetGeneratorOffset(
        self.ptr,
        offset,
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub fn reset_state(&mut self) -> CurandResult {
    let status = unsafe { curandGenerateSeeds(
        self.ptr,
    ) };
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_u32(&mut self, output: *mut u32, count: usize) -> CurandResult {
    let status = curandGenerate(
        self.ptr,
        output,
        count,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_u64(&mut self, output: *mut u64, count: usize) -> CurandResult {
    let status = curandGenerateLongLong(
        self.ptr,
        output,
        count,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_uniform_f32(&mut self, output: *mut f32, count: usize) -> CurandResult {
    let status = curandGenerateUniform(
        self.ptr,
        output,
        count,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_uniform_f64(&mut self, output: *mut f64, count: usize) -> CurandResult {
    let status = curandGenerateUniformDouble(
        self.ptr,
        output,
        count,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_normal_f32(&mut self, output: *mut f32, count: usize, mean: f32, stddev: f32) -> CurandResult {
    let status = curandGenerateNormal(
        self.ptr,
        output,
        count,
        mean,
        stddev,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_normal_f64(&mut self, output: *mut f64, count: usize, mean: f64, stddev: f64) -> CurandResult {
    let status = curandGenerateNormalDouble(
        self.ptr,
        output,
        count,
        mean,
        stddev,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_log_normal_f32(&mut self, output: *mut f32, count: usize, mean: f32, stddev: f32) -> CurandResult {
    let status = curandGenerateLogNormal(
        self.ptr,
        output,
        count,
        mean,
        stddev,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }

  pub unsafe fn gen_log_normal_f64(&mut self, output: *mut f64, count: usize, mean: f64, stddev: f64) -> CurandResult {
    let status = curandGenerateLogNormalDouble(
        self.ptr,
        output,
        count,
        mean,
        stddev,
    );
    match status {
      CURAND_STATUS_SUCCESS => Ok(()),
      _ => return Err(CurandError(status)),
    }
  }
}
