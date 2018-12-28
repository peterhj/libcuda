use crate::ffi::*;

use cuda_ffi_types::cuda::*;

use std::os::raw::{c_void};
use std::path::{PathBuf};
use std::ptr::{null_mut};

pub fn is_cuda_initialized() -> bool {
  let mut count: i32 = 0;
  let result = unsafe { cuDeviceGetCount(&mut count as *mut _) };
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

pub struct CuModule {
  ptr:  CUmodule,
}

impl CuModule {
  pub fn load<P: Into<PathBuf>>(file_path: P) -> CuResult<CuModule> {
    // TODO
    unimplemented!();
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

  pub fn unload(mut self) -> CuResult<()> {
    match unsafe { cuModuleUnload(self.ptr) } {
      cudaError_enum_CUDA_SUCCESS => Ok(()),
      e => Err(e),
    }
  }
}
