use ffi::runtime::{cudaStream_t};

use libc::{c_char, size_t};

#[repr(C)]
struct cudnnContext;
pub type cudnnHandle_t = *mut cudnnContext;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnStatus_t {
  Success         = 0,
  NotInitialized  = 1,
  AllocFailed     = 2,
  BadParam        = 3,
  InternalError   = 4,
  InvalidError    = 5,
  ArchMismatch    = 6,
  MappingError    = 7,
  ExecutionFailed = 8,
  NotSupported    = 9,
  LicenseError    = 10,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnDataType_t {
  Float   = 0,
  Double  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnTensorFormat_t {
  RowMajorNCHW    = 0,
  InterleavedNHWC = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnAddMode_t {
  Image       = 0,
  //SameHW      = 0,
  FeatureMap  = 1,
  //SameCHW     = 1,
  SameC       = 2,
  FullTensor  = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionMode_t {
  Convolution       = 0,
  CrossCorrelation  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionFwdPreference_t {
  NoWorkspace           = 0,
  PreferFastest         = 1,
  SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionFwdAlgo_t {
  ImplicitGemm        = 0,
  ImplicitPrecompGemm = 1,
  Gemm                = 2,
  Direct              = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxAlgorithm_t {
  Fast      = 0,
  Accurate  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxMode_t {
  Instance  = 0,
  Channel   = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnPoolingMode_t {
  Max                           = 0,
  AverageCountIncludingPadding  = 1,
  AverageCountExcludingPadding  = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnActivationMode_t {
  Sigmoid = 0,
  Relu    = 1,
  Tanh    = 2,
}

#[link(name = "cudnn", kind = "dylib")]
extern "C" {
  pub fn cudnnGetVersion() -> size_t;

  pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

  pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
  pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
  pub fn cudnnSetStream(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
  pub fn cudnnGetStream(handle: cudnnHandle_t, stream: *mut cudaStream_t) -> cudnnStatus_t;

  // TODO
}
