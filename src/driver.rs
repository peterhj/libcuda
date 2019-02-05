#![allow(non_upper_case_globals)]

use crate::ffi::cuda::*;

use std::ffi::{CStr};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::{null, null_mut};

pub fn get_version() -> CuResult<i32> {
  let mut version: c_int = -1;
  match unsafe { cuDriverGetVersion(&mut version as *mut c_int) } {
    cudaError_enum_CUDA_SUCCESS => Ok(version),
    e => Err(CuError(e)),
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CuError(pub CUresult);

impl fmt::Debug for CuError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(f, "CuError({}, name={}, desc={})",
        self.get_code(), self.get_name(), self.get_desc())
  }
}

impl CuError {
  pub fn get_code(&self) -> u32 {
    self.0 as u32
  }

  pub fn get_name(&self) -> &'static str {
    let mut sp: *const c_char = null();
    match unsafe { cuGetErrorName(self.0, &mut sp as *mut *const c_char) } {
      cudaError_enum_CUDA_SUCCESS => {}
      cudaError_enum_CUDA_ERROR_INVALID_VALUE => {
        return "(invalid CUresult)";
      }
      e => panic!("cuGetErrorName failed: {}", e),
    }
    assert!(!sp.is_null());
    let s: &'static CStr = unsafe { CStr::from_ptr(sp) };
    match s.to_str() {
      Err(_) => "(invalid utf-8)",
      Ok(s) => s,
    }
  }

  pub fn get_desc(&self) -> &'static str {
    let mut sp: *const c_char = null();
    match unsafe { cuGetErrorString(self.0, &mut sp as *mut *const c_char) } {
      cudaError_enum_CUDA_SUCCESS => {}
      cudaError_enum_CUDA_ERROR_INVALID_VALUE => {
        return "(invalid CUresult)";
      }
      e => panic!("cuGetErrorString failed: {}", e),
    }
    assert!(!sp.is_null());
    let s: &'static CStr = unsafe { CStr::from_ptr(sp) };
    match s.to_str() {
      Err(_) => "(invalid utf-8)",
      Ok(s) => s,
    }
  }
}

pub type CuResult<T=()> = Result<T, CuError>;

pub fn cuda_initialized() -> bool {
  let mut count: i32 = 0;
  let result = unsafe { cuDeviceGetCount(&mut count as *mut c_int) };
  match result {
    cudaError_enum_CUDA_SUCCESS => true,
    cudaError_enum_CUDA_ERROR_NOT_INITIALIZED => false,
    e => panic!("cuDeviceGetCount failed: {:?}", CuError(e)),
  }
}

pub fn cuda_init() {
  let result = unsafe { cuInit(0) };
  match result {
    cudaError_enum_CUDA_SUCCESS => {}
    e => panic!("cuInit failed: {:?}", CuError(e)),
  }
}

#[derive(Debug)]
pub struct CuModule {
  ptr:  CUmodule,
}

impl CuModule {
  pub fn load<P: AsRef<CStr>>(module_path: &P) -> CuResult<CuModule> {
    let module_path_cstr = module_path.as_ref();
    let mut ptr: CUmodule = null_mut();
    match unsafe { cuModuleLoad(
        &mut ptr as *mut CUmodule,
        module_path_cstr.as_ptr()) }
    {
      cudaError_enum_CUDA_SUCCESS => Ok(CuModule{ptr}),
      e => Err(CuError(e)),
    }
  }

  pub fn load_fat_binary_image_unchecked(image: &[u8]) -> CuResult<CuModule> {
    let mut ptr: CUmodule = null_mut();
    match unsafe { cuModuleLoadFatBinary(
        &mut ptr as *mut CUmodule,
        image.as_ptr() as *const c_void) }
    {
      cudaError_enum_CUDA_SUCCESS => Ok(CuModule{ptr}),
      e => Err(CuError(e)),
    }
  }

  pub fn unload(mut self) -> CuResult {
    assert!(!self.ptr.is_null());
    match unsafe { cuModuleUnload(self.ptr) } {
      cudaError_enum_CUDA_SUCCESS => {
        self.ptr = null_mut();
        Ok(())
      }
      e => Err(CuError(e)),
    }
  }

  pub fn as_raw(&self) -> CUmodule {
    self.ptr
  }
}
