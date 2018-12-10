extern crate bindgen;

use std::env;
use std::fs;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  let cuda_dir = PathBuf::from(match env::var("CUDA_HOME") {
    Ok(path) => path,
    Err(_) => "/usr/local/cuda".to_owned(),
  });

  println!("cargo:rustc-link-lib=cuda");
  println!("cargo:rustc-link-lib=cudart");

  fs::remove_file(out_dir.join("driver_bind.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_driver.h")
    .whitelist_recursively(false)
    .whitelist_var("__CUDA_API_VERSION")
    .whitelist_var("CUDA_VERSION")
    .whitelist_type("cudaError_enum")
    .whitelist_type("CUresult")
    .whitelist_type("CUdevice")
    .whitelist_type("CUdevice_attribute")
    .whitelist_type("CUdevice_attribute_enum")
    .whitelist_type("CUuuid_st")
    .whitelist_type("CUuuid")
    .whitelist_type("CUctx_st")
    .whitelist_type("CUcontext")
    .whitelist_type("CUstream_st")
    .whitelist_type("CUstream")
    .whitelist_type("CUevent_st")
    .whitelist_type("CUevent")
    .whitelist_type("CUdeviceptr")
    .whitelist_function("cuInit")
    .whitelist_function("cuDeviceGet")
    .whitelist_function("cuDeviceGetAttr")
    .whitelist_function("cuDeviceGetCount")
    .whitelist_function("cuDeviceGetName")
    .whitelist_function("cuDeviceGetUuid")
    .whitelist_function("cuDeviceTotalMem")
    .whitelist_function("cuDevicePrimaryCtxGetState")
    .whitelist_function("cuDevicePrimaryCtxRelease")
    .whitelist_function("cuDevicePrimaryCtxReset")
    .whitelist_function("cuDevicePrimaryCtxRetain")
    .whitelist_function("cuDevicePrimaryCtxSetFlags")
    .whitelist_function("cuCtxGetApiVersion")
    .whitelist_function("cuCtxGetCurrent")
    .whitelist_function("cuCtxGetDevice")
    .whitelist_function("cuStreamGetCtx")
    .generate()
    .expect("bindgen failed to generate driver bindings")
    .write_to_file(out_dir.join("driver_bind.rs"))
    .expect("bindgen failed to write driver bindings");

  fs::remove_file(out_dir.join("driver_types_bind.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_driver_types.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaError")
    .whitelist_type("cudaError_t")
    .whitelist_type("cudaDeviceAttr")
    .whitelist_type("cudaStream_t")
    .whitelist_type("cudaEvent_t")
    .whitelist_type("cudaMemoryAdvise")
    .whitelist_type("cudaMemcpyKind")
    .whitelist_type("cudaMemRangeAttribute")
    .whitelist_type("cudaGLDeviceList")
    .whitelist_type("cudaGraphicsResource")
    .whitelist_type("cudaGraphicsResource_t")
    .generate()
    .expect("bindgen failed to generate driver types bindings")
    .write_to_file(out_dir.join("driver_types_bind.rs"))
    .expect("bindgen failed to write driver types bindings");

  fs::remove_file(out_dir.join("runtime_bind.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_runtime.h")
    .whitelist_recursively(false)
    // Device management.
    .whitelist_type("cudaDeviceProp")
    .whitelist_function("cudaDeviceReset")
    .whitelist_function("cudaDeviceSynchronize")
    .whitelist_function("cudaGetDeviceCount")
    .whitelist_function("cudaGetDevice")
    .whitelist_function("cudaGetDeviceFlags")
    .whitelist_function("cudaGetDeviceProperties")
    .whitelist_function("cudaDeviceGetAttribute")
    .whitelist_function("cudaSetDevice")
    .whitelist_function("cudaSetDeviceFlags")
    // Error handling.
    .whitelist_function("cudaGetErrorString")
    // Stream management.
    .whitelist_type("cudaStreamCallback_t")
    .whitelist_function("cudaStreamCreate")
    .whitelist_function("cudaStreamCreateWithFlags")
    .whitelist_function("cudaStreamCreateWithPriority")
    .whitelist_function("cudaStreamDestroy")
    .whitelist_function("cudaStreamAddCallback")
    .whitelist_function("cudaStreamAttachMemAsync")
    .whitelist_function("cudaStreamQuery")
    .whitelist_function("cudaStreamSynchronize")
    .whitelist_function("cudaStreamWaitEvent")
    // Event management.
    .whitelist_function("cudaEventCreate")
    .whitelist_function("cudaEventCreateWithFlags")
    .whitelist_function("cudaEventDestroy")
    .whitelist_function("cudaEventElapsedTime")
    .whitelist_function("cudaEventQuery")
    .whitelist_function("cudaEventRecord")
    .whitelist_function("cudaEventSynchronize")
    // Memory management.
    .whitelist_function("cudaMalloc")
    .whitelist_function("cudaFree")
    .whitelist_function("cudaMallocHost")
    .whitelist_function("cudaFreeHost")
    .whitelist_function("cudaMallocManaged")
    .whitelist_function("cudaMemAdvise")
    .whitelist_function("cudaMemPrefetchAsync")
    .whitelist_function("cudaMemRangeGetAttribute")
    .whitelist_function("cudaMemRangeGetAttributes")
    .whitelist_function("cudaMemcpy")
    .whitelist_function("cudaMemcpyAsync")
    .whitelist_function("cudaMemcpy2D")
    .whitelist_function("cudaMemcpy2DAsync")
    .whitelist_function("cudaMemcpyPeer")
    .whitelist_function("cudaMemcpyPeerAsync")
    .whitelist_function("cudaMemset")
    .whitelist_function("cudaMemsetAsync")
    // Peer device memory access.
    .whitelist_function("cudaDeviceCanAccessPeer")
    .whitelist_function("cudaDeviceDisablePeerAccess")
    .whitelist_function("cudaDeviceEnablePeerAccess")
    // OpenGL interoperability.
    .whitelist_function("cudaGLGetDevices")
    .whitelist_function("cudaGraphicsGLRegisterBuffer")
    .whitelist_function("cudaGraphicsGLRegisterImage")
    // Graphics interoperability.
    .whitelist_function("cudaGraphicsMapResources")
    .whitelist_function("cudaGraphicsResourceGetMappedPointer")
    .whitelist_function("cudaGraphicsResourceSetMapFlags")
    .whitelist_function("cudaGraphicsUnmapResources")
    .whitelist_function("cudaGraphicsUnregisterResource")
    .generate()
    .expect("bindgen failed to generate runtime bindings")
    .write_to_file(out_dir.join("runtime_bind.rs"))
    .expect("bindgen failed to write runtime bindings");

  fs::remove_file(out_dir.join("libtypes_bind.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_libtypes.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaDataType")
    .whitelist_type("cudaDataType_t")
    .generate()
    .expect("bindgen failed to generate libtypes bindings")
    .write_to_file(out_dir.join("libtypes_bind.rs"))
    .expect("bindgen failed to write libtypes bindings");
}
