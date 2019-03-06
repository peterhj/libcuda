#[cfg(feature = "fresh")]
extern crate bindgen;

use std::env;
#[cfg(feature = "fresh")]
use std::fs;
use std::path::{PathBuf};

#[cfg(all(
    not(feature = "fresh"),
    not(any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    ))
))]
fn main() {
  compile_error!("a cuda version feature must be enabled");
}

fn to_cuda_lib_dir(cuda_dir: &PathBuf) -> PathBuf {
  if cfg!(target_os = "linux") {
    if cfg!(target_arch = "x86_64") {
      cuda_dir.join("lib64")
    } else if cfg!(target_arch = "powerpc64le") {
      panic!("todo: ppc64le support on linux is not yet implemented");
    } else {
      panic!("unsupported target arch on linux");
    }
  } else if cfg!(target_os = "windows") {
    if cfg!(target_arch = "x86_64") {
      cuda_dir.join("lib").join("x64")
    } else {
      panic!("unsupported target arch on windows");
    }
  } else if cfg!(target_os = "macos") {
    unimplemented!();
  } else {
    panic!("unsupported target os");
  }
}

#[cfg(all(
    not(feature = "fresh"),
    any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    )
))]
fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
  println!("cargo:rustc-link-lib=cuda");
  println!("cargo:rustc-link-lib=cudart");
  println!("cargo:rustc-link-lib=curand");
  println!("cargo:rustc-link-lib=cublas");
  let maybe_cuda_dir =
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok().map(|s| PathBuf::from(s));
  let maybe_cuda_fallback_lib_dir =
      maybe_cuda_dir.as_ref().map(|d| to_cuda_lib_dir(d));
  let maybe_cuda_lib_dir =
      env::var("CUDA_LIBRARY_PATH")
        .ok().map(|s| PathBuf::from(s));
  if let Some(cuda_lib_dir) = maybe_cuda_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_fallback_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
}

#[cfg(feature = "fresh")]
fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
  println!("cargo:rustc-link-lib=cuda");
  println!("cargo:rustc-link-lib=cudart");
  println!("cargo:rustc-link-lib=curand");
  println!("cargo:rustc-link-lib=cublas");

  let maybe_cuda_dir =
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok().map(|s| PathBuf::from(s));
  let maybe_cuda_fallback_lib_dir =
      maybe_cuda_dir.as_ref().map(|d| to_cuda_lib_dir(d));
  let maybe_cuda_lib_dir =
      env::var("CUDA_LIBRARY_PATH")
        .ok().map(|s| PathBuf::from(s));

  if let Some(cuda_lib_dir) = maybe_cuda_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_fallback_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }

  let maybe_cuda_include_dir =
      maybe_cuda_dir.as_ref().map(|d| d.join("include"));

  #[cfg(feature = "cuda_6_5")]
  let a_cuda_version_feature_must_be_enabled = "v6_5";
  #[cfg(feature = "cuda_7_0")]
  let a_cuda_version_feature_must_be_enabled = "v7_0";
  #[cfg(feature = "cuda_7_5")]
  let a_cuda_version_feature_must_be_enabled = "v7_5";
  #[cfg(feature = "cuda_8_0")]
  let a_cuda_version_feature_must_be_enabled = "v8_0";
  #[cfg(feature = "cuda_9_0")]
  let a_cuda_version_feature_must_be_enabled = "v9_0";
  #[cfg(feature = "cuda_9_1")]
  let a_cuda_version_feature_must_be_enabled = "v9_1";
  #[cfg(feature = "cuda_9_2")]
  let a_cuda_version_feature_must_be_enabled = "v9_2";
  #[cfg(feature = "cuda_10_0")]
  let a_cuda_version_feature_must_be_enabled = "v10_0";
  let v = a_cuda_version_feature_must_be_enabled;

  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let gensrc_dir = manifest_dir.join("gensrc").join("ffi").join(v);
  fs::create_dir_all(&gensrc_dir).ok();

  fs::remove_file(gensrc_dir.join("_cuda.rs")).ok();
  let builder = bindgen::Builder::default();
  if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  }
    .header("wrapped_cuda.h")
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
    .whitelist_type("CUstreamCallback")
    .whitelist_type("CUevent_st")
    .whitelist_type("CUevent")
    .whitelist_type("CUdeviceptr")
    .whitelist_type("CUjit_option_enum")
    .whitelist_type("CUjit_option")
    .whitelist_type("CUjitInputType_enum")
    .whitelist_type("CUjitInputType")
    .whitelist_type("CUlinkState_st")
    .whitelist_type("CUlinkState")
    .whitelist_type("CUmod_st")
    .whitelist_type("CUmodule")
    .whitelist_type("CUhostFn")
    .whitelist_type("CUfunc_st")
    .whitelist_type("CUfunction")
    .whitelist_type("CUmem_advise_enum")
    .whitelist_type("CUmem_advise")
    .whitelist_type("CUmem_range_attribute_enum")
    .whitelist_type("CUmem_range_attribute")
    .whitelist_type("CUpointer_attribute_enum")
    .whitelist_type("CUpointer_attribute")
    .whitelist_type("CUsurfref_st")
    .whitelist_type("CUsurfref")
    .whitelist_type("CUtexref_st")
    .whitelist_type("CUtexref")
    .whitelist_type("CUarray_st")
    .whitelist_type("CUarray")
    .whitelist_type("CUDA_LAUNCH_PARAMS_st")
    .whitelist_type("CUDA_LAUNCH_PARAMS")
    .whitelist_function("cuGetErrorName")
    .whitelist_function("cuGetErrorString")
    .whitelist_function("cuInit")
    .whitelist_function("cuDriverGetVersion")
    .whitelist_function("cuDeviceGet")
    .whitelist_function("cuDeviceGetAttribute")
    .whitelist_function("cuDeviceGetCount")
    .whitelist_function("cuDeviceGetLuid")
    .whitelist_function("cuDeviceGetName")
    .whitelist_function("cuDeviceGetUuid")
    .whitelist_function("cuDeviceTotalMem")
    .whitelist_function("cuDevicePrimaryCtxGetState")
    .whitelist_function("cuDevicePrimaryCtxRelease")
    .whitelist_function("cuDevicePrimaryCtxReset")
    .whitelist_function("cuDevicePrimaryCtxRetain")
    .whitelist_function("cuDevicePrimaryCtxSetFlags")
    .whitelist_function("cuCtxCreate_v2")
    .whitelist_function("cuCtxDestroy_v2")
    .whitelist_function("cuCtxGetApiVersion")
    //.whitelist_function("cuCtxGetCacheConfig")
    .whitelist_function("cuCtxGetCurrent")
    .whitelist_function("cuCtxGetDevice")
    .whitelist_function("cuCtxGetFlags")
    //.whitelist_function("cuCtxGetLimit")
    //.whitelist_function("cuCtxGetSharedMemConfig")
    .whitelist_function("cuCtxGetStreamPriorityRange")
    .whitelist_function("cuCtxPopCurrent_v2")
    .whitelist_function("cuCtxPushCurrent_v2")
    //.whitelist_function("cuCtxSetCacheConfig")
    .whitelist_function("cuCtxSetCurrent")
    //.whitelist_function("cuCtxSetLimit")
    //.whitelist_function("cuCtxSetSharedMemConfig")
    .whitelist_function("cuCtxSynchronize")
    .whitelist_function("cuLinkAddData_v2")
    .whitelist_function("cuLinkAddFile_v2")
    .whitelist_function("cuLinkComplete")
    .whitelist_function("cuLinkCreate_v2")
    .whitelist_function("cuLinkDestroy")
    .whitelist_function("cuModuleGetFunction")
    .whitelist_function("cuModuleGetGlobal")
    .whitelist_function("cuModuleGetSurfRef")
    .whitelist_function("cuModuleGetTexRef")
    .whitelist_function("cuModuleLoad")
    .whitelist_function("cuModuleLoadData")
    .whitelist_function("cuModuleLoadDataEx")
    .whitelist_function("cuModuleLoadFatBinary")
    .whitelist_function("cuModuleUnload")
    .whitelist_function("cuMemsetD16_v2")
    .whitelist_function("cuMemsetD16Async")
    .whitelist_function("cuMemsetD32_v2")
    .whitelist_function("cuMemsetD32Async")
    .whitelist_function("cuMemsetD8_v2")
    .whitelist_function("cuMemsetD8Async")
    .whitelist_function("cuMemAdvise")
    .whitelist_function("cuMemPrefetchAsync")
    .whitelist_function("cuMemRangeGetAttribute")
    .whitelist_function("cuMemRangeGetAttributes")
    .whitelist_function("cuPointerGetAttribute")
    .whitelist_function("cuPointerGetAttributes")
    .whitelist_function("cuPointerSetAttribute")
    .whitelist_function("cuStreamGetCtx")
    //.whitelist_function("cuStreamBatchMemOp")
    //.whitelist_function("cuStreamWaitValue32")
    //.whitelist_function("cuStreamWaitValue64")
    //.whitelist_function("cuStreamWriteValue32")
    //.whitelist_function("cuStreamWriteValue64")
    //.whitelist_function("cuFuncGetAttribute")
    //.whitelist_function("cuFuncSetAttribute")
    //.whitelist_function("cuFuncSetCacheConfig")
    //.whitelist_function("cuFuncSetSharedMemConfig")
    .whitelist_function("cuLaunchCooperativeKernel")
    .whitelist_function("cuLaunchCooperativeKernelMultiDevice")
    .whitelist_function("cuLaunchHostFunc")
    .whitelist_function("cuLaunchKernel")
    .generate_comments(false)
    .prepend_enum_name(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate driver bindings")
    .write_to_file(gensrc_dir.join("_cuda.rs"))
    .expect("bindgen failed to write driver bindings");

  if cfg!(feature = "cuda_gte_9_0") {
    fs::remove_file(gensrc_dir.join("_cuda_fp16.rs")).ok();
    let builder = bindgen::Builder::default();
    if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
      builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
    } else {
      builder
    }
      .clang_arg("-x").clang_arg("c++")
      .clang_arg("-std=c++11")
      .header("wrapped_cuda_fp16.h")
      // NB: `whitelist_recursively(false)` and `derive_copy(true)` still
      // not compatible, see:
      // https://github.com/rust-lang/rust-bindgen/issues/1454
      //.whitelist_recursively(false)
      .derive_copy(true)
      .whitelist_type("__half")
      .whitelist_type("__half2")
      .generate_comments(false)
      .rustfmt_bindings(true)
      .generate()
      .expect("bindgen failed to generate fp16 bindings")
      .write_to_file(gensrc_dir.join("_cuda_fp16.rs"))
      .expect("bindgen failed to write fp16 bindings");
  }

  fs::remove_file(gensrc_dir.join("_cuda_runtime_api.rs")).ok();
  let builder = bindgen::Builder::default();
  if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  }
    .header("wrapped_cuda_runtime_api.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaStreamCallback_t")
    // Device management.
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
    // Occupancy.
    .whitelist_function("cudaOccupancyMaxActiveBlocksPerMultiprocessor")
    .whitelist_function("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
    // Memory management.
    .whitelist_function("cudaMalloc")
    .whitelist_function("cudaFree")
    .whitelist_function("cudaMallocHost")
    .whitelist_function("cudaFreeHost")
    .whitelist_function("cudaHostAlloc")
    .whitelist_function("cudaHostGetDevicePointer")
    .whitelist_function("cudaHostGetFlags")
    .whitelist_function("cudaHostRegister")
    .whitelist_function("cudaHostUnregister")
    .whitelist_function("cudaMallocManaged")
    .whitelist_function("cudaMemAdvise")
    .whitelist_function("cudaMemGetInfo")
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
    // Version management.
    .whitelist_function("cudaDriverGetVersion")
    .whitelist_function("cudaRuntimeGetVersion")
    .generate_comments(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate runtime bindings")
    .write_to_file(gensrc_dir.join("_cuda_runtime_api.rs"))
    .expect("bindgen failed to write runtime bindings");

  fs::remove_file(gensrc_dir.join("_driver_types.rs")).ok();
  let builder = bindgen::Builder::default();
  if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  }
    .header("wrapped_driver_types.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaError")
    .whitelist_type("cudaError_t")
    .whitelist_type("cudaDeviceAttr")
    .whitelist_type("cudaUUID_t")
    .whitelist_type("cudaDeviceProp")
    .whitelist_type("cudaStream_t")
    .whitelist_type("cudaEvent_t")
    .whitelist_type("cudaMemoryAdvise")
    .whitelist_type("cudaMemcpyKind")
    .whitelist_type("cudaMemRangeAttribute")
    .whitelist_type("cudaGLDeviceList")
    .whitelist_type("cudaGraphicsResource")
    .whitelist_type("cudaGraphicsResource_t")
    .generate_comments(false)
    .prepend_enum_name(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate driver types bindings")
    .write_to_file(gensrc_dir.join("_driver_types.rs"))
    .expect("bindgen failed to write driver types bindings");

  if cfg!(feature = "cuda_gte_8_0") {
    fs::remove_file(gensrc_dir.join("_library_types.rs")).ok();
    let builder = bindgen::Builder::default();
    if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
      builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
    } else {
      builder
    }
      .header("wrapped_library_types.h")
      .whitelist_recursively(false)
      .whitelist_type("cudaDataType")
      .whitelist_type("cudaDataType_t")
      .whitelist_type("libraryPropertyType")
      .whitelist_type("libraryPropertyType_t")
      .generate_comments(false)
      .prepend_enum_name(false)
      .rustfmt_bindings(true)
      .generate()
      .expect("bindgen failed to generate library types bindings")
      .write_to_file(gensrc_dir.join("_library_types.rs"))
      .expect("bindgen failed to write library types bindings");
  }

  fs::remove_file(gensrc_dir.join("_curand.rs")).ok();
  let builder = bindgen::Builder::default();
  if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  }
    .header("wrapped_curand.h")
    .whitelist_recursively(false)
    .whitelist_type("curandStatus")
    .whitelist_type("curandStatus_t")
    .whitelist_type("curandGenerator_st")
    .whitelist_type("curandGenerator_t")
    .whitelist_type("curandRngType")
    .whitelist_type("curandRngType_t")
    .whitelist_type("curandDiscreteDistribution_st")
    .whitelist_type("curandDiscreteDistribution_t")
    .whitelist_type("curandDirectionVectors32_t")
    .whitelist_type("curandDirectionVectors64_t")
    .whitelist_type("curandDirectionVectorSet")
    .whitelist_type("curandDirectionVectorSet_t")
    .whitelist_type("curandOrdering")
    .whitelist_type("curandOrdering_t")
    .whitelist_function("curandCreateGenerator")
    .whitelist_function("curandCreateGeneratorHost")
    .whitelist_function("curandCreatePoissonDistribution")
    .whitelist_function("curandDestroyDistribution")
    .whitelist_function("curandDestroyGenerator")
    .whitelist_function("curandGenerate")
    .whitelist_function("curandGenerateLogNormal")
    .whitelist_function("curandGenerateLogNormalDouble")
    .whitelist_function("curandGenerateLongLong")
    .whitelist_function("curandGenerateNormal")
    .whitelist_function("curandGenerateNormalDouble")
    .whitelist_function("curandGeneratePoisson")
    .whitelist_function("curandGenerateSeeds")
    .whitelist_function("curandGenerateUniform")
    .whitelist_function("curandGenerateUniformDouble")
    .whitelist_function("curandGetDirectionVectors32")
    .whitelist_function("curandGetDirectionVectors64")
    .whitelist_function("curandGetProperty")
    .whitelist_function("curandGetScrambleConstants32")
    .whitelist_function("curandGetScrambleConstants64")
    .whitelist_function("curandGetStream")
    .whitelist_function("curandGetVersion")
    .whitelist_function("curandSetGeneratorOffset")
    .whitelist_function("curandSetGeneratorOrdering")
    .whitelist_function("curandSetPseudoRandomGeneratorSeed")
    .whitelist_function("curandSetQuasiRandomGeneratorDimensions")
    .whitelist_function("curandSetStream")
    .generate_comments(false)
    .prepend_enum_name(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate curand bindings")
    .write_to_file(gensrc_dir.join("_curand.rs"))
    .expect("bindgen failed to write curand bindings");

  fs::remove_file(gensrc_dir.join("_cublas.rs")).ok();
  let builder = bindgen::Builder::default();
  if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  }
    .header("wrapped_cublas.h")
    .whitelist_recursively(false)
    .whitelist_type("cublasStatus")
    .whitelist_type("cublasStatus_t")
    .whitelist_type("cublasContext")
    .whitelist_type("cublasHandle_t")
    .whitelist_type("cublasPointerMode")
    .whitelist_type("cublasPointerMode_t")
    .whitelist_type("cublasAtomicsMode")
    .whitelist_type("cublasAtomicsMode_t")
    .whitelist_type("cublasMath")
    .whitelist_type("cublasMath_t")
    .whitelist_type("cublasLogCallback")
    .whitelist_type("cublasOperation")
    .whitelist_type("cublasOperation_t")
    .whitelist_type("cublasGemmAlgo")
    .whitelist_type("cublasGemmAlgo_t")
    .whitelist_function("cublasCreate_v2")
    .whitelist_function("cublasDestroy_v2")
    .whitelist_function("cublasGetVersion_v2")
    .whitelist_function("cublasGetProperty")
    .whitelist_function("cublasSetStream_v2")
    .whitelist_function("cublasGetStream_v2")
    .whitelist_function("cublasSetPointerMode_v2")
    .whitelist_function("cublasGetPointerMode_v2")
    .whitelist_function("cublasSetAtomicsMode")
    .whitelist_function("cublasGetAtomicsMode")
    .whitelist_function("cublasSetMathMode")
    .whitelist_function("cublasGetMathMode")
    .whitelist_function("cublasLoggerConfigure")
    .whitelist_function("cublasGetLoggerCallback")
    .whitelist_function("cublasSetLoggerCallback")
    .whitelist_function("cublasSaxpy_v2")
    .whitelist_function("cublasSdot_v2")
    .whitelist_function("cublasSnrm2_v2")
    .whitelist_function("cublasSscal_v2")
    .whitelist_function("cublasDaxpy_v2")
    .whitelist_function("cublasDdot_v2")
    .whitelist_function("cublasDnrm2_v2")
    .whitelist_function("cublasDscal_v2")
    .whitelist_function("cublasSgemv_v2")
    .whitelist_function("cublasDgemv_v2")
    .whitelist_function("cublasSgemm_v2")
    .whitelist_function("cublasDgemm_v2")
    .whitelist_function("cublasHgemm")
    .whitelist_function("cublasSgemmEx")
    .whitelist_function("cublasGemmEx")
    .generate_comments(false)
    .prepend_enum_name(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate cublas bindings")
    .write_to_file(gensrc_dir.join("_cublas.rs"))
    .expect("bindgen failed to write cublas bindings");
}
