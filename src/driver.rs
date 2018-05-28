use ::ffi::driver::*;

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
