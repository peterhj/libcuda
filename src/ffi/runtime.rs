#![allow(dead_code)]
#![allow(missing_copy_implementations)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libc::{
  c_void, c_char, c_int, c_uint, c_ulonglong, c_float, c_double, size_t,
};

pub const CUDART_VERSION: c_int = 6050;

#[repr(C)]
pub struct dim3 {
  x: c_uint,
  y: c_uint,
  z: c_uint,
}

enum cudaArray {}
pub type cudaArray_t = *mut cudaArray;
pub type cudaArray_const_t = *mut cudaArray;

pub type cudaError_t = cudaError;

enum CUevent_st {}
pub type cudaEvent_t = *mut CUevent_st;

enum cudaGraphicsResource {}
pub type cudaGraphicsResource_t = *mut cudaGraphicsResource;

enum cudaMipmappedArray {}
pub type cudaMipmappedArray_t = *mut cudaMipmappedArray;
pub type cudaMipmappedArray_const_t = *mut cudaMipmappedArray;

pub type cudaOutputMode_t = cudaOutputMode;

enum CUstream_st {}
pub type cudaStream_t = *mut CUstream_st;

pub type cudaSurfaceObject_t = c_ulonglong;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[repr(C)]
pub enum cudaError {
  Success                      =      0,
  MissingConfiguration         =      1,
  MemoryAllocation             =      2,
  InitializationError          =      3,
  LaunchFailure                =      4,
  PriorLaunchFailure           =      5,
  LaunchTimeout                =      6,
  LaunchOutOfResources         =      7,
  InvalidDeviceFunction        =      8,
  InvalidConfiguration         =      9,
  InvalidDevice                =     10,
  InvalidValue                 =     11,
  InvalidPitchValue            =     12,
  InvalidSymbol                =     13,
  MapBufferObjectFailed        =     14,
  UnmapBufferObjectFailed      =     15,
  InvalidHostPointer           =     16,
  InvalidDevicePointer         =     17,
  InvalidTexture               =     18,
  InvalidTextureBinding        =     19,
  InvalidChannelDescriptor     =     20,
  InvalidMemcpyDirection       =     21,
  AddressOfConstant            =     22,
  TextureFetchFailed           =     23,
  TextureNotBound              =     24,
  SynchronizationError         =     25,
  InvalidFilterSetting         =     26,
  InvalidNormSetting           =     27,
  MixedDeviceExecution         =     28,
  CudartUnloading              =     29,
  Unknown                      =     30,
  NotYetImplemented            =     31,
  MemoryValueTooLarge          =     32,
  InvalidResourceHandle        =     33,
  NotReady                     =     34,
  InsufficientDriver           =     35,
  SetOnActiveProcess           =     36,
  InvalidSurface               =     37,
  NoDevice                     =     38,
  ECCUncorrectable             =     39,
  SharedObjectSymbolNotFound   =     40,
  SharedObjectInitFailed       =     41,
  UnsupportedLimit             =     42,
  DuplicateVariableName        =     43,
  DuplicateTextureName         =     44,
  DuplicateSurfaceName         =     45,
  DevicesUnavailable           =     46,
  InvalidKernelImage           =     47,
  NoKernelImageForDevice       =     48,
  IncompatibleDriverContext    =     49,
  PeerAccessAlreadyEnabled     =     50,
  PeerAccessNotEnabled         =     51,
  DeviceAlreadyInUse           =     54,
  ProfilerDisabled             =     55,
  ProfilerNotInitialized       =     56,
  ProfilerAlreadyStarted       =     57,
  ProfilerAlreadyStopped       =     58,
  Assert                       =     59,
  TooManyPeers                 =     60,
  HostMemoryAlreadyRegistered  =     61,
  HostMemoryNotRegistered      =     62,
  OperatingSystem              =     63,
  PeerAccessUnsupported        =     64,
  LaunchMaxDepthExceeded       =     65,
  LaunchFileScopedTex          =     66,
  LaunchFileScopedSurf         =     67,
  SyncDepthExceeded            =     68,
  LaunchPendingCountExceeded   =     69,
  NotPermitted                 =     70,
  NotSupported                 =     71,
  HardwareStackError           =     72,
  IllegalInstruction           =     73,
  MisalignedAddress            =     74,
  InvalidAddressSpace          =     75,
  InvalidPc                    =     76,
  IllegalAddress               =     77,
  InvalidPtx                   =     78,
  InvalidGraphicsContext       =     79,
  StartupFailure               =   0x7f,
  ApiFailureBase               =  10000,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaDeviceFlags {
  ScheduleAuto          = 0x00,
  ScheduleSpin          = 0x01,
  ScheduleYield         = 0x02,
  ScheduleBlockingSync  = 0x04,
  MapHost               = 0x08,
  LmemResizeToMax       = 0x10,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaChannelFormatKind {
  Signed = 0,
  Unsigned = 1,
  Float = 2,
  None = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaComputeMode {
  Default = 0,
  Exclusive = 1,
  Prohibited = 2,
  ExclusiveProcess = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaDeviceAttr {
  MaxThreadsPerBlock             = 1,
  MaxBlockDimX                   = 2,
  MaxBlockDimY                   = 3,
  MaxBlockDimZ                   = 4,
  MaxGridDimX                    = 5,
  MaxGridDimY                    = 6,
  MaxGridDimZ                    = 7,
  MaxSharedMemoryPerBlock        = 8,
  TotalConstantMemory            = 9,
  WarpSize                       = 10,
  MaxPitch                       = 11,
  MaxRegistersPerBlock           = 12,
  ClockRate                      = 13,
  TextureAlignment               = 14,
  GpuOverlap                     = 15,
  MultiProcessorCount            = 16,
  KernelExecTimeout              = 17,
  Integrated                     = 18,
  CanMapHostMemory               = 19,
  ComputeMode                    = 20,
  MaxTexture1DWidth              = 21,
  MaxTexture2DWidth              = 22,
  MaxTexture2DHeight             = 23,
  MaxTexture3DWidth              = 24,
  MaxTexture3DHeight             = 25,
  MaxTexture3DDepth              = 26,
  MaxTexture2DLayeredWidth       = 27,
  MaxTexture2DLayeredHeight      = 28,
  MaxTexture2DLayeredLayers      = 29,
  SurfaceAlignment               = 30,
  ConcurrentKernels              = 31,
  EccEnabled                     = 32,
  PciBusId                       = 33,
  PciDeviceId                    = 34,
  TccDriver                      = 35,
  MemoryClockRate                = 36,
  GlobalMemoryBusWidth           = 37,
  L2CacheSize                    = 38,
  MaxThreadsPerMultiProcessor    = 39,
  AsyncEngineCount               = 40,
  UnifiedAddressing              = 41,    
  MaxTexture1DLayeredWidth       = 42,
  MaxTexture1DLayeredLayers      = 43,
  MaxTexture2DGatherWidth        = 45,
  MaxTexture2DGatherHeight       = 46,
  MaxTexture3DWidthAlt           = 47,
  MaxTexture3DHeightAlt          = 48,
  MaxTexture3DDepthAlt           = 49,
  PciDomainId                    = 50,
  TexturePitchAlignment          = 51,
  MaxTextureCubemapWidth         = 52,
  MaxTextureCubemapLayeredWidth  = 53,
  MaxTextureCubemapLayeredLayers = 54,
  MaxSurface1DWidth              = 55,
  MaxSurface2DWidth              = 56,
  MaxSurface2DHeight             = 57,
  MaxSurface3DWidth              = 58,
  MaxSurface3DHeight             = 59,
  MaxSurface3DDepth              = 60,
  MaxSurface1DLayeredWidth       = 61,
  MaxSurface1DLayeredLayers      = 62,
  MaxSurface2DLayeredWidth       = 63,
  MaxSurface2DLayeredHeight      = 64,
  MaxSurface2DLayeredLayers      = 65,
  MaxSurfaceCubemapWidth         = 66,
  MaxSurfaceCubemapLayeredWidth  = 67,
  MaxSurfaceCubemapLayeredLayers = 68,
  MaxTexture1DLinearWidth        = 69,
  MaxTexture2DLinearWidth        = 70,
  MaxTexture2DLinearHeight       = 71,
  MaxTexture2DLinearPitch        = 72,
  MaxTexture2DMipmappedWidth     = 73,
  MaxTexture2DMipmappedHeight    = 74,
  ComputeCapabilityMajor         = 75, 
  ComputeCapabilityMinor         = 76,
  MaxTexture1DMipmappedWidth     = 77,
  StreamPrioritiesSupported      = 78,
  GlobalL1CacheSupported         = 79,
  LocalL1CacheSupported          = 80,
  MaxSharedMemoryPerMultiprocessor = 81,
  MaxRegistersPerMultiprocessor  = 82,
  ManagedMemory                  = 83,
  IsMultiGpuBoard                = 84,
  MultiGpuBoardGroupID           = 85,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaFuncCache {
  PreferNone = 0,
  PreferShared = 1,
  PreferL1 = 2,
  PreferEqual = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaGraphicsCubeFace {
  PositiveX = 0x00,
  NegativeX = 0x01,
  PositiveY = 0x02,
  NegativeY = 0x03,
  PositiveZ = 0x04,
  NegativeZ = 0x05,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaGraphicsMapFlags {
  None = 0,
  ReadOnly = 1,
  WriteDiscard = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaGraphicsRegisterFlags {
  None = 0,
  ReadOnly = 1,
  WriteDiscard = 2,
  SurfaceLoadStore = 4,
  TextureGather = 8,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaLimit {
  StackSize = 0x00,
  PrintfFifoSize = 0x01,
  MallocHeapSize = 0x02,
  DevRuntimeSyncDepth = 0x03,
  DevRuntimePendingLaunchCount = 0x04,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaMemcpyKind {
  HostToHost = 0,
  HostToDevice = 1,
  DeviceToHost = 2,
  DeviceToDevice = 3,
  Default = 4,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaMemoryType {
  Host = 1,
  Device = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaOutputMode {
  KeyValuePair = 0x00,
  CSV = 0x01,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaResourceType {
  Array = 0x00,
  MipmappedArray = 0x01,
  Linear = 0x02,
  Pitch2D = 0x03,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaResourceViewFormat {
  None                      = 0x00,
  UnsignedChar1             = 0x01,
  UnsignedChar2             = 0x02,
  UnsignedChar4             = 0x03,
  SignedChar1               = 0x04,
  SignedChar2               = 0x05,
  SignedChar4               = 0x06,
  UnsignedShort1            = 0x07,
  UnsignedShort2            = 0x08,
  UnsignedShort4            = 0x09,
  SignedShort1              = 0x0a,
  SignedShort2              = 0x0b,
  SignedShort4              = 0x0c,
  UnsignedInt1              = 0x0d,
  UnsignedInt2              = 0x0e,
  UnsignedInt4              = 0x0f,
  SignedInt1                = 0x10,
  SignedInt2                = 0x11,
  SignedInt4                = 0x12,
  Half1                     = 0x13,
  Half2                     = 0x14,
  Half4                     = 0x15,
  Float1                    = 0x16,
  Float2                    = 0x17,
  Float4                    = 0x18,
  UnsignedBlockCompressed1  = 0x19,
  UnsignedBlockCompressed2  = 0x1a,
  UnsignedBlockCompressed3  = 0x1b,
  UnsignedBlockCompressed4  = 0x1c,
  SignedBlockCompressed4    = 0x1d,
  UnsignedBlockCompressed5  = 0x1e,
  SignedBlockCompressed5    = 0x1f,
  UnsignedBlockCompressed6H = 0x20,
  SignedBlockCompressed6H   = 0x21,
  UnsignedBlockCompressed7  = 0x22,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaSharedMemConfig {
  BankSizeDefault = 0,
  BankSizeFourByte = 1,
  BankSizeEightByte = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaSurfaceBoundaryMode {
  Zero = 0,
  Clamp = 1,
  Trap = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaSurfaceFormatMode {
  Forced = 0,
  Auto = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaTextureAddressMode {
  Wrap = 0,
  Clamp = 1,
  Mirror = 2,
  Border = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaTextureFilterMode {
  Point = 0,
  Linear = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudaTextureReadMode {
  ElementType = 0,
  NormalizedFloat = 1,
}

#[repr(C)]
pub struct cudaChannelFormatDesc {
  x: c_int,
  y: c_int,
  z: c_int,
  w: c_int,
  f: cudaChannelFormatKind,
}

#[repr(C)]
pub struct cudaDeviceProp {
  name: [c_char; 256],
  totalGlobalMem: size_t,
  sharedMemPerBlock: size_t,
  regsPerBlock: c_int,
  warpSize: c_int,
  memPitch: size_t,
  maxThreadsPerBlock: c_int,
  maxThreadsDim: [c_int; 3],
  maxGridSize: [c_int; 3],
  clockRate: c_int,
  totalConstMem: size_t,
  major: c_int,
  minor: c_int,
  textureAlignment: size_t,
  texturePitchAlignment: size_t,
  deviceOverlap: c_int,
  multiProcessorCount: c_int,
  kernelExecTimeoutEnabled: c_int,
  integrated: c_int,
  canMapHostMemory: c_int,
  computeMode: c_int,
  maxTexture1D: c_int,
  maxTexture1DMipmap: c_int,
  maxTexture1DLinear: c_int,
  maxTexture2D: [c_int; 2],
  maxTexture2DMipmap: [c_int; 2],
  maxTexture2DLinear: [c_int; 3],
  maxTexture2DGather: [c_int; 2],
  maxTexture3D: [c_int; 3],
  maxTexture3DAlt: [c_int; 3],
  maxTextureCubemap: c_int,
  maxTexture1DLayered: [c_int; 2],
  maxTexture2DLayered: [c_int; 3],
  maxTextureCubemapLayered: [c_int; 2],
  maxSurface1D: c_int,
  maxSurface2D: [c_int; 2],
  maxSurface3D: [c_int; 3],
  maxSurface1DLayered: [c_int; 2],
  maxSurface2DLayered: [c_int; 3],
  maxSurfaceCubemap: c_int,
  maxSurfaceCubemapLayered: [c_int; 2],
  surfaceAlignment: size_t,
  concurrentKernels: c_int,
  ECCEnabled: c_int,
  pciBusID: c_int,
  pciDeviceID: c_int,
  pciDomainID: c_int,
  tccDriver: c_int,
  asyncEngineCount: c_int,
  unifiedAddressing: c_int,
  memoryClockRate: c_int,
  memoryBusWidth: c_int,
  l2CacheSize: c_int,
  maxThreadsPerMultiProcessor: c_int,
  streamPrioritiesSupported: c_int,
  globalL1CacheSupported: c_int,
  localL1CacheSupported: c_int,
  sharedMemPerMultiprocessor: size_t,
  regsPerMultiprocessor: c_int,
  managedMemory: c_int,
  isMultiGpuBoard: c_int,
  multiGpuBoardGroupID: c_int,
}

#[repr(C)]
pub struct cudaExtent {
  width: size_t,
  height: size_t,
  depth: size_t,
}

#[repr(C)]
pub struct cudaFuncAttributes {
  sharedSizeBytes: size_t,
  constSizeBytes: size_t,
  localSizeBytes: size_t,
  maxThreadsPerBlock: c_int,
  numRegs: c_int,
  ptxVersion: c_int,
  binaryVersion: c_int,
  cacheModeCA: c_int,
}

pub const CUDA_IPC_HANDLE_SIZE: c_int = 64;

#[repr(C)]
pub struct cudaIpcEventHandle_t {
  reserved: [c_char; CUDA_IPC_HANDLE_SIZE as usize],
}

#[repr(C)]
pub struct cudaIpcMemHandle_t {
  reserved: [c_char; CUDA_IPC_HANDLE_SIZE as usize],
}

#[repr(C)]
pub struct cudaMemcpy3DParms {
  srcArray: cudaArray_t,
  srcPos: cudaPos,
  srcPtr: cudaPitchedPtr,
  dstArray: cudaArray_t,
  dstPos: cudaPos,
  dstPtr: cudaPitchedPtr,
  extent: cudaExtent,
  kind: cudaMemcpyKind,
}

#[repr(C)]
pub struct cudaMemcpy3DPeerParms {
  srcArray: cudaArray_t,
  srcPos: cudaPos,
  srcPtr: cudaPitchedPtr,
  srcDevice: c_int,
  dstArray: cudaArray_t,
  dstPos: cudaPos,
  dstPtr: cudaPitchedPtr,
  dstDevice: c_int,
  extent: cudaExtent,
}

#[repr(C)]
pub struct cudaPitchedPtr {
  ptr: *mut c_void,
  pitch: size_t,
  xsize: size_t,
  ysize: size_t,
}

#[repr(C)]
pub struct cudaPointerAttributes {
  memoryType: cudaMemoryType,
  device: c_int,
  devicePointer: *mut c_void,
  hostPointer: *mut c_void,
  isManaged: c_int,
}

#[repr(C)]
pub struct cudaPos {
  x: size_t,
  y: size_t,
  z: size_t,
}

#[repr(C)]
pub struct cudaResourceDescUnion {
  devPtr: *mut c_void,
  desc: cudaChannelFormatDesc,
  width: size_t,
  height: size_t,
  pitchInBytes: size_t,
}

#[repr(C)]
pub struct cudaResourceDesc {
  resType: cudaResourceType,
  // TODO(20150115): `res` requires FFI unions:
  // <https://github.com/rust-lang/rust/issues/5492>
  // (see: cuda/include/driver_types.h:954 for the original).
  res: cudaResourceDescUnion,
}

#[repr(C)]
pub struct cudaResourceViewDesc {
  format: cudaResourceViewFormat,
  width: size_t,
  height: size_t,
  depth: size_t,
  firstMipmapLevel: c_uint,
  lastMipmapLevel: c_uint,
  firstLayer: c_uint,
  lastLayer: c_uint,
}

#[repr(C)]
pub struct cudaTextureDesc {
  addressMode: [cudaTextureAddressMode; 3],
  filterMode: cudaTextureFilterMode,
  readMode: cudaTextureReadMode,
  sRGB: c_int,
  normalizedCoords: c_int,
  maxAnisotropy: c_uint,
  mipmapFilterMode: cudaTextureFilterMode,
  mipmapLevelBias: c_float,
  minMipmapLevelClamp: c_float,
  maxMipmapLevelClamp: c_float,
}

#[repr(C)]
pub struct surfaceReference {
  channelDesc: cudaChannelFormatDesc,
}

#[repr(C)]
pub struct textureReference {
  normalized: c_int,
  filterMode: cudaTextureFilterMode,
  addressMode: [cudaTextureAddressMode; 3],
  channelDesc: cudaChannelFormatDesc,
  sRGB: c_int,
  maxAnisotropy: c_uint,
  mipmapFilterMode: cudaTextureFilterMode,
  mipmapLevelBias: c_float,
  minMipmapLevelClamp: c_float,
  maxMipmapLevelClamp: c_float,
  __cudaReserved: [c_int; 15],
}

pub type cudaStreamCallback_t = extern "C" fn (cudaStream_t, cudaError_t, *mut c_void);

//#[link(name = "cudart", kind = "dylib")]
#[link(name = "cudart")]
extern "C" {
  // Error Handling
  pub fn cudaGetErrorName(error: cudaError_t) -> *mut c_char;
  pub fn cudaGetErrorString(error: cudaError_t) -> *mut c_char;
  pub fn cudaGetLastError() -> cudaError_t;
  pub fn cudaPeekAtLastError() -> cudaError_t;

  // Version Management
  pub fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;
  pub fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t;

  // Device Management
  pub fn cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> cudaError_t;
  pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> cudaError_t;
  pub fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> cudaError_t;
  pub fn cudaDeviceGetCacheConfig(pCacheConfig: *mut *mut cudaFuncCache) -> cudaError_t;
  pub fn cudaDeviceGetLimit(pValue: *mut size_t, limit: cudaLimit) -> cudaError_t;
  pub fn cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> cudaError_t;
  pub fn cudaDeviceGetSharedMemConfig(pConfig: *mut *mut cudaSharedMemConfig) -> cudaError_t;
  pub fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> cudaError_t;
  pub fn cudaDeviceReset() -> cudaError_t;
  pub fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
  pub fn cudaDeviceSetLimit(limit: cudaLimit, value: size_t) -> cudaError_t;
  pub fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;
  pub fn cudaDeviceSynchronize() -> cudaError_t;
  pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
  pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
  pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
  pub fn cudaIpcCloseMemHandle() -> cudaError_t;
  pub fn cudaIpcGetEventHandle() -> cudaError_t;
  pub fn cudaIpcGetMemHandle() -> cudaError_t;
  pub fn cudaIpcOpenEventHandle() -> cudaError_t;
  pub fn cudaIpcOpenMemHandle() -> cudaError_t;
  pub fn cudaSetDevice(device: c_int) -> cudaError_t;
  pub fn cudaSetDeviceFlags(flags: c_uint) -> cudaError_t;
  pub fn cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> cudaError_t;

  // Stream Management
  pub fn cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userdata: *mut c_void, flags: c_uint) -> cudaError_t;
  pub fn cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: *mut c_void, length: size_t, flags: c_uint) -> cudaError_t;
  pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
  pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
  pub fn cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: c_uint, priority: c_int) -> cudaError_t;
  pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
  pub fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> cudaError_t;
  pub fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> cudaError_t;
  pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
  pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
  pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t;

  // Event Management
  pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
  pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
  pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
  pub fn cudaEventElapsedTime(ms: *mut c_float, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
  pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
  pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
  pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;

  // Execution Control
  pub fn cudaConfigureCall(gridDim: dim3, blockDim: dim3, sharedMem: size_t, stream: cudaStream_t) -> cudaError_t;
  pub fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t;
  pub fn cudaFuncSetCacheConfig(func: *const c_void, cacheConfig: cudaFuncCache) -> cudaError_t;
  pub fn cudaFuncSetSharedMemConfig(func: *const c_void, config: cudaSharedMemConfig) -> cudaError_t;
  pub fn cudaLaunch(func: *const c_void) -> cudaError_t;
  pub fn cudaSetDoubleForDevice(d: *mut c_double) -> cudaError_t;
  pub fn cudaSetDoubleForHost(d: *mut c_double) -> cudaError_t;
  pub fn cudaSetupArgument(arg: *const c_void, size: size_t, offset: size_t) -> cudaError_t;

  // Occupancy
  // ...

  // Memory Management
  pub fn cudaArrayGetInfo(desc: *mut cudaChannelFormatDesc, extent: *mut cudaExtent, flags: *mut c_uint, array: cudaArray_t) -> cudaError_t;
  pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
  pub fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;
  pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
  pub fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;
  pub fn cudaGetMipmappedArrayLevel(levelArray: *mut cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: c_uint) -> cudaError_t;
  pub fn cudaGetSymbolAddress(devPtr: *mut *mut c_void, symbol: *const c_void) -> cudaError_t;
  pub fn cudaGetSymbolSize(size: *mut size_t, symbol: *const c_void) -> cudaError_t;
  pub fn cudaHostAlloc(pHost: *mut *mut c_void, size: size_t, flags: c_uint) -> cudaError_t;
  pub fn cudaHostGetDevicePointer(pDevice: *mut *mut c_void, pHost: *mut c_void, flags: c_uint) -> cudaError_t;
  pub fn cudaHostGetFlags(pFlags: *mut c_uint, pHost: *mut c_void) -> cudaError_t;
  pub fn cudaHostRegister(ptr: *mut c_void, size: size_t, flags: c_uint) -> cudaError_t;
  pub fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;
  pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError_t;
  pub fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;
  pub fn cudaMalloc3DArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, flags: c_uint) -> cudaError_t;
  pub fn cudaMallocArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, width: size_t, height: size_t, flags: c_uint) -> cudaError_t;
  pub fn cudaMallocHost(ptr: *mut *mut c_void, size: size_t) -> cudaError_t;
  pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: size_t, flags: c_uint) -> cudaError_t;
  pub fn cudaMallocMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, numLevels: c_uint, flags: c_uint) -> cudaError_t;
  pub fn cudaMallocPitch(devPtr: *mut *mut c_void, pitch: *mut size_t, width: size_t, height: size_t) -> cudaError_t;
  pub fn cudaMemGetInfo(free: *mut size_t, total: *mut size_t) -> cudaError_t;
  pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: size_t, kind: cudaMemcpyKind) -> cudaError_t;
  pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: size_t, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
  // ...
  pub fn cudaMemcpy2D(dst: *mut c_void, dpitch: size_t, src: *const c_void, spitch: size_t, width: size_t, height: size_t, kind: cudaMemcpyKind) -> cudaError_t;
  pub fn cudaMemcpy2DAsync(dst: *mut c_void, dpitch: size_t, src: *const c_void, spitch: size_t, width: size_t, height: size_t, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
  // ...
  //pub fn cudaMemcpy3D() -> cudaError_t;
  // ...
  pub fn cudaMemcpyPeer(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: size_t) -> cudaError_t;
  pub fn cudaMemcpyPeerAsync(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: size_t, stream: cudaStream_t) -> cudaError_t;
  // ...
  pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: size_t) -> cudaError_t;
  // ...
  pub fn cudaMemsetAsync(devPtr: *mut c_void, value: c_int, count: size_t, stream: cudaStream_t) -> cudaError_t;
  pub fn make_cudaExtent(w: size_t, h: size_t, d: size_t) -> cudaExtent;
  pub fn make_cudaPitchedPtr(d: *mut c_void, p: size_t, xsz: size_t, ysz: size_t) -> cudaPitchedPtr;
  pub fn make_cudaPos(x: size_t, y: size_t, z: size_t) -> cudaPos;

  // Unified Addressing
  // ...

  // Peer Device Memory Access
  pub fn cudaDeviceCanAccessPeer(canAccessPeer: *mut c_int, device: c_int, peer_device: c_int) -> cudaError_t;
  pub fn cudaDeviceDisablePeerAccess(peerDevice: c_int) -> cudaError_t;
  pub fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;
}
