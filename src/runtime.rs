#![allow(non_upper_case_globals)]

use crate::extras::{MemInfo};
use crate::ffi::cuda_runtime_api::*;
use crate::ffi::driver_types::*;

use std::ffi::{CStr};
use std::mem::{size_of, zeroed};
use std::os::raw::{c_void, c_int, c_uint};
use std::ptr::{null_mut};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudaError(pub cudaError_t);

impl CudaError {
  pub fn get_code(&self) -> u32 {
    let &CudaError(e) = self;
    e as _
  }

  pub fn get_string(&self) -> String {
    let raw_s = unsafe { cudaGetErrorString(self.0) };
    if raw_s.is_null() {
      return format!("(null)");
    }
    let cs = unsafe { CStr::from_ptr(raw_s) };
    let s = match cs.to_str() {
      Err(_) => "(invalid utf-8)",
      Ok(s) => s,
    };
    s.to_owned()
  }
}

pub type CudaResult<T=()> = Result<T, CudaError>;

pub fn get_driver_version() -> CudaResult<i32> {
  let mut version: c_int = -1;
  match unsafe { cudaDriverGetVersion(&mut version as *mut c_int) } {
    cudaSuccess => {
      assert!(version >= 0);
      Ok(version)
    }
    e => Err(CudaError(e)),
  }
}

pub fn get_runtime_version() -> CudaResult<i32> {
  let mut version: c_int = -1;
  match unsafe { cudaRuntimeGetVersion(&mut version as *mut c_int) } {
    cudaSuccess => {
      assert!(version >= 0);
      Ok(version)
    }
    e => Err(CudaError(e)),
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudaDevice(pub i32);

impl CudaDevice {
  /// Count the number of devices.
  ///
  /// Corresponds to `cudaGetDeviceCount`.
  pub fn count() -> CudaResult<i32> {
    let mut count: c_int = 0;
    match unsafe { cudaGetDeviceCount(&mut count as *mut c_int) } {
      cudaSuccess => {
        assert!(count >= 0);
        Ok(count)
      }
      e => Err(CudaError(e)),
    }
  }

  /// Reset the current device.
  ///
  /// Corresponds to `cudaDeviceReset`.
  pub fn reset_current() -> CudaResult {
    match unsafe { cudaDeviceReset() } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Synchronize all work on the current device.
  ///
  /// Corresponds to `cudaDeviceSynchronize`.
  pub fn synchronize_current() -> CudaResult {
    match unsafe { cudaDeviceSynchronize() } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Set flags for the current device.
  ///
  /// Corresponds to `cudaSetDeviceFlags`.
  pub fn set_flags_current(flags: u32) -> CudaResult {
    match unsafe { cudaSetDeviceFlags(flags as c_uint) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Query the current device.
  ///
  /// Corresponds to `cudaGetDevice`.
  pub fn get_current() -> CudaResult<CudaDevice> {
    let mut curr_dev: c_int = 0;
    match unsafe { cudaGetDevice(&mut curr_dev as *mut c_int) } {
      cudaSuccess => Ok(CudaDevice(curr_dev)),
      e => Err(CudaError(e)),
    }
  }

  /// Set the current device.
  ///
  /// Corresponds to `cudaSetDevice`.
  pub fn set_current(&self) -> CudaResult {
    match unsafe { cudaSetDevice(self.0 as c_int) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Query the `cudaDeviceProp` properties struct for the given device.
  ///
  /// Corresponds to `cudaGetDeviceProperties`.
  pub fn get_properties(&self) -> CudaResult<cudaDeviceProp> {
    let mut prop: cudaDeviceProp = unsafe { zeroed() };
    match unsafe { cudaGetDeviceProperties(&mut prop as *mut cudaDeviceProp, self.0 as c_int) } {
      cudaSuccess => Ok(prop),
      e => Err(CudaError(e)),
    }
  }

  /// Query the given attribute for the given device.
  ///
  /// Corresponds to `cudaGetDeviceAttribute`.
  pub fn get_attribute(&self, attr: cudaDeviceAttr) -> CudaResult<i32> {
    let mut value: c_int = 0;
    match unsafe { cudaDeviceGetAttribute(&mut value as *mut c_int, attr, self.0 as c_int) } {
      cudaSuccess => Ok(value as i32),
      e => Err(CudaError(e)),
    }
  }

  /// Check whether peer device access to `peer_dev` can be enabled.
  ///
  /// Corresponds to `cudaDeviceCanAccessPeer`.
  pub fn can_access_peer(&self, peer_dev: i32) -> CudaResult<bool> {
    let mut access: c_int = 0;
    match unsafe { cudaDeviceCanAccessPeer(&mut access as *mut c_int, self.0 as c_int, peer_dev as c_int) } {
      cudaSuccess => Ok(access != 0),
      e => Err(CudaError(e)),
    }
  }

  /// Enable peer device access from the current device to `peer_dev`.
  /// Returns whether or not peer device access was previously enabled.
  ///
  /// Corresponds to `cudaDeviceEnablePeerAccess`.
  pub fn enable_peer_access_current(peer_dev: i32) -> CudaResult<bool> {
    match unsafe { cudaDeviceEnablePeerAccess(peer_dev as c_int, 0) } {
      cudaSuccess => Ok(false),
      cudaErrorPeerAccessAlreadyEnabled => Ok(true),
      e => Err(CudaError(e)),
    }
  }

  /// Disable peer device access from the current device to `peer_dev`.
  /// Returns whether or not peer device access was previously enabled.
  ///
  /// Corresponds to `cudaDeviceDisablePeerAccess`.
  pub fn disable_peer_access_current(peer_dev: i32) -> CudaResult<bool> {
    match unsafe { cudaDeviceDisablePeerAccess(peer_dev as c_int) } {
      cudaSuccess => Ok(true),
      cudaErrorPeerAccessNotEnabled => Ok(false),
      e => Err(CudaError(e)),
    }
  }

  /// Returns the free and total device memory in bytes for the current device.
  ///
  /// Corresponds to `cudaMemGetInfo`.
  pub fn get_mem_info_current() -> CudaResult<MemInfo> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    match unsafe { cudaMemGetInfo(&mut free as *mut _, &mut total as *mut _) } {
      cudaSuccess => Ok(MemInfo{free, total}),
      e => Err(CudaError(e)),
    }
  }
}

#[derive(Debug)]
pub struct CudaStream {
  ptr:  cudaStream_t,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      match unsafe { cudaStreamDestroy(self.ptr) } {
        cudaSuccess => {}
        cudaErrorCudartUnloading => {
          // NB(20160308): Sometimes drop() is called while the global runtime
          // is shutting down; suppress these errors.
        }
        e => {
          let err = CudaError(e);
          panic!("FATAL: CudaStream::drop() failed: {:?} ({})",
              err, err.get_string());
        }
      }
    }
  }
}

impl CudaStream {
  pub fn default() -> CudaStream {
    CudaStream{ptr: null_mut()}
  }

  pub fn create() -> CudaResult<CudaStream> {
    let mut ptr: cudaStream_t = null_mut();
    match unsafe { cudaStreamCreate(&mut ptr as *mut cudaStream_t) } {
      cudaSuccess => Ok(CudaStream{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub fn as_raw(&self) -> cudaStream_t {
    self.ptr
  }

  pub fn ptr_eq(&self, other: &CudaStream) -> bool {
    self.ptr == other.ptr
  }

  pub fn add_callback(&mut self, callback: extern "C" fn (stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void), user_data: *mut c_void) -> CudaResult {
    match unsafe { cudaStreamAddCallback(self.ptr, Some(callback), user_data, 0) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn synchronize(&mut self) -> CudaResult {
    match unsafe { cudaStreamSynchronize(self.ptr) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn wait_event(&mut self, event: &mut CudaEvent) -> CudaResult {
    match unsafe { cudaStreamWaitEvent(self.ptr, event.as_raw(), 0) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e))
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaEventStatus {
  Complete,
  NotReady,
}

#[derive(Debug)]
pub struct CudaEvent {
  ptr:  cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      match unsafe { cudaEventDestroy(self.ptr) } {
        cudaSuccess => {}
        cudaErrorCudartUnloading => {
          // NB(20160308): Sometimes drop() is called while the global runtime
          // is shutting down; suppress these errors.
        }
        e => {
          let err = CudaError(e);
          panic!("FATAL: CudaEvent::drop() failed: {:?} ({})",
              err, err.get_string());
        }
      }
    }
  }
}

impl CudaEvent {
  pub fn create() -> CudaResult<CudaEvent> {
    let mut ptr = null_mut() as cudaEvent_t;
    match unsafe { cudaEventCreate(&mut ptr as *mut cudaEvent_t) } {
      cudaSuccess => Ok(CudaEvent{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub fn blocking() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x01)
  }

  pub fn fastest() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x02)
  }

  pub fn create_with_flags(flags: u32) -> CudaResult<CudaEvent> {
    let mut ptr = null_mut() as cudaEvent_t;
    match unsafe { cudaEventCreateWithFlags(&mut ptr as *mut cudaEvent_t, flags) } {
      cudaSuccess => Ok(CudaEvent{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub fn as_raw(&self) -> cudaEvent_t {
    self.ptr
  }

  pub fn ptr_eq(&self, other: &CudaEvent) -> bool {
    self.ptr == other.ptr
  }

  pub fn query(&mut self) -> CudaResult<CudaEventStatus> {
    match unsafe { cudaEventQuery(self.ptr) } {
      cudaSuccess => Ok(CudaEventStatus::Complete),
      cudaErrorNotReady => Ok(CudaEventStatus::NotReady),
      e => Err(CudaError(e)),
    }
  }

  pub fn record(&mut self, stream: &mut CudaStream) -> CudaResult {
    match unsafe { cudaEventRecord(self.ptr, stream.as_raw()) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn synchronize(&mut self) -> CudaResult {
    match unsafe { cudaEventSynchronize(self.ptr) } {
      cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }
}

pub fn cuda_alloc_device(size: usize) -> CudaResult<*mut u8> {
  let mut dptr: *mut c_void = null_mut();
  match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
    cudaSuccess => Ok(dptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub fn cuda_alloc_host(size: usize) -> CudaResult<*mut u8> {
  let mut ptr: *mut c_void = null_mut();
  match unsafe { cudaMallocHost(&mut ptr as *mut *mut c_void, size) } {
    cudaSuccess => Ok(ptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub fn cuda_alloc_host_with_flags(size: usize, flags: u32) -> CudaResult<*mut u8> {
  let mut ptr: *mut c_void = null_mut();
  match unsafe { cudaHostAlloc(&mut ptr as *mut *mut c_void, size, flags) } {
    cudaSuccess => Ok(ptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_device(dptr: *mut u8) -> CudaResult {
  match cudaFree(dptr as *mut c_void) {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_host(ptr: *mut u8) -> CudaResult {
  match cudaFreeHost(ptr as *mut c_void) {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset(dptr: *mut u8, value: i32, size: usize) -> CudaResult {
  match cudaMemset(dptr as *mut c_void, value, size) {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset_async(dptr: *mut u8, value: i32, size: usize, stream: &mut CudaStream) -> CudaResult {
  match cudaMemsetAsync(dptr as *mut c_void, value, size, stream.as_raw()) {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaMemcpyKind {
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  Unified,
}

impl CudaMemcpyKind {
  pub fn to_raw(&self) -> cudaMemcpyKind {
    match *self {
      CudaMemcpyKind::HostToHost      => cudaMemcpyHostToHost,
      CudaMemcpyKind::HostToDevice    => cudaMemcpyHostToDevice,
      CudaMemcpyKind::DeviceToHost    => cudaMemcpyDeviceToHost,
      CudaMemcpyKind::DeviceToDevice  => cudaMemcpyDeviceToDevice,
      CudaMemcpyKind::Unified         => cudaMemcpyDefault,
    }
  }
}

pub unsafe fn cuda_memcpy<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind) -> CudaResult
where T: Copy + 'static
{
  match cudaMemcpy(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw())
  {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_async<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind,
    stream: &mut CudaStream) -> CudaResult
where T: Copy + 'static
{
  match cudaMemcpyAsync(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw(),
      stream.as_raw())
  {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_2d_async<T>(
    dst: *mut T,
    dst_pitch_bytes: usize,
    src: *const T,
    src_pitch_bytes: usize,
    width: usize,
    height: usize,
    kind: CudaMemcpyKind,
    stream: &mut CudaStream) -> CudaResult
where T: Copy + 'static
{
  let width_bytes = width * size_of::<T>();
  assert!(width_bytes <= dst_pitch_bytes);
  assert!(width_bytes <= src_pitch_bytes);
  match cudaMemcpy2DAsync(
      dst as *mut c_void,
      dst_pitch_bytes,
      src as *const c_void,
      src_pitch_bytes,
      width_bytes,
      height,
      kind.to_raw(),
      stream.as_raw())
  {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_peer_async<T>(
    dst: *mut T,
    dst_device_idx: i32,
    src: *const T,
    src_device_idx: i32,
    len: usize,
    stream: &mut CudaStream) -> CudaResult
where T: Copy + 'static
{
  match cudaMemcpyPeerAsync(
      dst as *mut c_void,
      dst_device_idx,
      src as *const c_void,
      src_device_idx,
      len * size_of::<T>(),
      stream.as_raw())
  {
    cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}
