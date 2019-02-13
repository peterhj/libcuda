#![allow(non_upper_case_globals)]

pub use cuda_runtime_api as driver_types;

pub mod cublas {
  pub use cuda_sys::cublas::*;

  pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = cublasStatus_t::SUCCESS;
  pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = cublasStatus_t::NOT_INITIALIZED;
  pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = cublasStatus_t::ALLOC_FAILED;
  pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = cublasStatus_t::INVALID_VALUE;
  pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = cublasStatus_t::ARCH_MISMATCH;
  pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = cublasStatus_t::MAPPING_ERROR;
  pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = cublasStatus_t::EXECUTION_FAILED;
  pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = cublasStatus_t::INTERNAL_ERROR;
  pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = cublasStatus_t::NOT_SUPPORTED;
  pub const CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = cublasStatus_t::LICENSE_ERROR;

  pub const CUBLAS_POINTER_MODE_HOST: cublasPointerMode_t = cublasPointerMode_t_CUBLAS_POINTER_MODE_HOST;
  pub const CUBLAS_POINTER_MODE_DEVICE: cublasPointerMode_t = cublasPointerMode_t_CUBLAS_POINTER_MODE_DEVICE;

  pub const CUBLAS_ATOMICS_NOT_ALLOWED: cublasAtomicsMode_t = cublasAtomicsMode_t_CUBLAS_ATOMICS_NOT_ALLOWED;
  pub const CUBLAS_ATOMICS_ALLOWED: cublasAtomicsMode_t = cublasAtomicsMode_t_CUBLAS_ATOMICS_ALLOWED;

  pub const CUBLAS_OP_N: cublasOperation_t = cublasOperation_t_CUBLAS_OP_N;
  pub const CUBLAS_OP_T: cublasOperation_t = cublasOperation_t_CUBLAS_OP_T;
  pub const CUBLAS_OP_C: cublasOperation_t = cublasOperation_t_CUBLAS_OP_C;
}

pub mod cuda {
  pub use cuda_sys::cuda::*;

  pub const CUDA_SUCCESS: cudaError_t = cudaError_t::CUDA_SUCCESS;
  pub const CUDA_ERROR_INVALID_VALUE: cudaError_t = cudaError_t::CUDA_ERROR_INVALID_VALUE;
  pub const CUDA_ERROR_NOT_INITIALIZED: cudaError_t = cudaError_t::CUDA_ERROR_NOT_INITIALIZED;
}

pub mod cuda_runtime_api {
  pub use cuda_sys::cudart::*;

  pub const cudaSuccess: cudaError_t = cudaError_t::Success;
  pub const cudaErrorPeerAccessAlreadyEnabled: cudaError_t = cudaError_t::PeerAccessAlreadyEnabled;
  pub const cudaErrorPeerAccessNotEnabled: cudaError_t = cudaError_t::PeerAccessNotEnabled;
  pub const cudaErrorCudartUnloading: cudaError_t = cudaError_t::CudartUnloading;
  pub const cudaErrorNotReady: cudaError_t = cudaError_t::NotReady;

  pub const cudaMemcpyHostToHost: cudaMemcpyKind = cudaMemcpyKind_cudaMemcpyHostToHost;
  pub const cudaMemcpyHostToDevice: cudaMemcpyKind = cudaMemcpyKind_cudaMemcpyHostToDevice;
  pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = cudaMemcpyKind_cudaMemcpyDeviceToHost;
  pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = cudaMemcpyKind_cudaMemcpyDeviceToDevice;
  pub const cudaMemcpyDefault: cudaMemcpyKind = cudaMemcpyKind_cudaMemcpyDefault;
}

const_assert_eq!(cuda_api_version; self::cuda::__CUDA_API_VERSION,  8000);
const_assert_eq!(cuda_version;     self::cuda::CUDA_VERSION,        8000);
