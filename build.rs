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

  fs::remove_file(out_dir.join("cuda.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_cuda.h")
    .whitelist_recursively(false)
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
    .whitelist_function("cuModuleGetFunction")
    .whitelist_function("cuModuleGetGlobal")
    .whitelist_function("cuModuleLoad")
    .whitelist_function("cuModuleLoadData")
    .whitelist_function("cuModuleLoadDataEx")
    .whitelist_function("cuModuleLoadFatBinary")
    .whitelist_function("cuModuleUnload")
    .whitelist_function("cuStreamGetCtx")
    .whitelist_function("cuLaunchCooperativeKernel")
    .whitelist_function("cuLaunchCooperativeKernelMultiDevice")
    .whitelist_function("cuLaunchHostFunc")
    .whitelist_function("cuLaunchKernel")
    .generate()
    .expect("bindgen failed to generate driver bindings")
    .write_to_file(out_dir.join("cuda.rs"))
    .expect("bindgen failed to write driver bindings");
}
