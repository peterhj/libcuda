extern crate bindgen;

use std::env;
use std::path::{PathBuf};

fn main() {
  //println!("cargo:rustc-link-lib=cudart");
  let cuda_bindings = bindgen::Builder::default()
    //.header("wrap_cuda.h")
    .header("/usr/local/cuda-7.5/include/cuda_runtime.h")
    .link("cudart")
    // Device management.
    .whitelisted_type("cudaDeviceProp")
    .whitelisted_function("cudaDeviceReset")
    .whitelisted_function("cudaDeviceSynchronize")
    .whitelisted_function("cudaGetDeviceCount")
    .whitelisted_function("cudaGetDevice")
    .whitelisted_function("cudaGetDeviceFlags")
    .whitelisted_function("cudaGetDeviceProperties")
    .whitelisted_function("cudaDeviceGetAttribute")
    .whitelisted_function("cudaSetDevice")
    .whitelisted_function("cudaSetDeviceFlags")
    // Error handling.
    .whitelisted_type("cudaError_t")
    // Stream management.
    .whitelisted_type("cudaStream_t")
    .whitelisted_function("cudaStreamCreate")
    .whitelisted_function("cudaStreamCreateWithFlags")
    .whitelisted_function("cudaStreamCreateWithPriority")
    .whitelisted_function("cudaStreamDestroy")
    .whitelisted_function("cudaStreamAddCallback")
    .whitelisted_function("cudaStreamSynchronize")
    .whitelisted_function("cudaStreamWaitEvent")
    // Event management.
    .whitelisted_type("cudaEvent_t")
    .whitelisted_function("cudaEventCreate")
    .whitelisted_function("cudaEventCreateWithFlags")
    .whitelisted_function("cudaEventDestroy")
    .whitelisted_function("cudaEventQuery")
    .whitelisted_function("cudaEventRecord")
    .whitelisted_function("cudaEventSynchronize")
    // Memory management.
    .whitelisted_function("cudaMalloc")
    .whitelisted_function("cudaFree")
    .whitelisted_function("cudaMallocHost")
    .whitelisted_function("cudaFreeHost")
    .whitelisted_function("cudaMemcpy")
    .whitelisted_function("cudaMemcpyAsync")
    .whitelisted_function("cudaMemcpyPeerAsync")
    .whitelisted_function("cudaMemset")
    .whitelisted_function("cudaMemsetAsync")
    // Peer device memory access.
    .whitelisted_function("cudaDeviceCanAccessPeer")
    .whitelisted_function("cudaDeviceDisablePeerAccess")
    .whitelisted_function("cudaDeviceEnablePeerAccess")
    .generate()
    .expect("bindgen failed to generate cuda bindings");
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  cuda_bindings
    .write_to_file(out_dir.join("cuda_bind.rs"))
    .expect("bindgen failed to write cuda bindings");
}
