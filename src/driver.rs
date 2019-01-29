#![allow(non_upper_case_globals)]

use crate::ffi::cuda::*;

use std::ffi::{CStr};
use std::os::raw::{c_int, c_void};
use std::ptr::{null_mut};

pub fn is_cuda_initialized() -> bool {
  let mut count: i32 = 0;
  let result = unsafe { cuDeviceGetCount(&mut count as *mut c_int) };
  match result {
    cudaError_enum_CUDA_SUCCESS => true,
    cudaError_enum_CUDA_ERROR_NOT_INITIALIZED => false,
    e => panic!("FATAL: cuDeviceGetCount failed: {}", e),
  }
}

pub fn cuda_init() {
  let result = unsafe { cuInit(0) };
  match result {
    cudaError_enum_CUDA_SUCCESS => {}
    e => panic!("FATAL: cuInit failed: {}", e),
  }
}

pub type CuResult<T> = Result<T, CUresult>;

pub fn get_version() -> CuResult<i32> {
  let mut version: c_int = -1;
  match unsafe { cuDriverGetVersion(&mut version as *mut c_int) } {
    cudaError_enum_CUDA_SUCCESS => Ok(version),
    e => Err(e),
  }
}

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
      e => Err(e),
    }
  }

  pub fn load_fat_binary_image(image: &[u8]) -> CuResult<CuModule> {
    let mut ptr: CUmodule = null_mut();
    match unsafe { cuModuleLoadFatBinary(
        &mut ptr as *mut CUmodule,
        image.as_ptr() as *const c_void) }
    {
      cudaError_enum_CUDA_SUCCESS => Ok(CuModule{ptr}),
      e => Err(e),
    }
  }

  pub fn unload(self) -> CuResult<()> {
    match unsafe { cuModuleUnload(self.ptr) } {
      cudaError_enum_CUDA_SUCCESS => Ok(()),
      e => Err(e),
    }
  }
}
