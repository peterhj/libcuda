use ::ffi::driver::*;

pub fn is_cuda_initialized() -> bool {
  let mut flags: u32 = 0;
  let mut active: i32 = 0;
  let result = unsafe { cuDevicePrimaryCtxGetState(0, &mut flags as *mut _, &mut active as *mut _) };
  match result {
    cudaError_enum_CUDA_SUCCESS => true,
    cudaError_enum_CUDA_ERROR_NOT_INITIALIZED => false,
    e => panic!("FATAL: cuInit failed: {}", e),
  }
}

pub fn cuda_init() {
  let result = unsafe { cuInit(0) };
  match result {
    cudaError_enum_CUDA_SUCCESS => {}
    e => panic!("FATAL: cuInit failed: {}", e),
  }
}
