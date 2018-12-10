use ffi::driver_types::*;
use ffi::runtime::*;

use std::ffi::{CStr};
use std::mem::{size_of, zeroed};
use std::os::raw::{c_void, c_int, c_uint};
use std::ptr::{null_mut};

#[derive(Clone, Copy, Debug)]
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
      Err(_) => "(invalid utf8)",
      Ok(s) => s,
    };
    s.to_owned()
  }
}

pub type CudaResult<T> = Result<T, CudaError>;

pub struct CudaDevice;

impl CudaDevice {
  pub fn count() -> CudaResult<usize> {
    let mut count: c_int = 0;
    unsafe {
      match cudaGetDeviceCount(&mut count as *mut c_int) {
        cudaError_cudaSuccess => Ok(count as usize),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn get_current() -> CudaResult<i32> {
    let mut index: c_int = 0;
    match unsafe { cudaGetDevice(&mut index as *mut c_int) } {
      cudaError_cudaSuccess => Ok(index),
      e => Err(CudaError(e)),
    }
  }

  pub fn set_current(index: i32) -> CudaResult<()> {
    unsafe {
      match cudaSetDevice(index as c_int) {
        cudaError_cudaSuccess => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn reset() -> CudaResult<()> {
    match unsafe { cudaDeviceReset() } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn synchronize() -> CudaResult<()> {
    match unsafe { cudaDeviceSynchronize() } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn set_flags(flags: u32) -> CudaResult<()> {
    match unsafe { cudaSetDeviceFlags(flags as c_uint) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn get_properties(device_idx: usize) -> CudaResult<cudaDeviceProp> {
    let mut prop: cudaDeviceProp = unsafe { zeroed() };
    match unsafe { cudaGetDeviceProperties(&mut prop as *mut _, device_idx as _) } {
      cudaError_cudaSuccess => Ok(prop),
      e => Err(CudaError(e)),
    }
  }

  pub fn get_attribute(device_idx: usize, attr: cudaDeviceAttr) -> CudaResult<i32> {
    let mut value: c_int = 0;
    match unsafe { cudaDeviceGetAttribute(&mut value as *mut c_int, attr, device_idx as c_int) } {
      cudaError_cudaSuccess => Ok(value as i32),
      e => Err(CudaError(e)),
    }
  }

  pub fn can_access_peer(idx: usize, peer_idx: usize) -> CudaResult<bool> {
    unsafe {
      let mut access: c_int = 0;
      match cudaDeviceCanAccessPeer(&mut access as *mut c_int, idx as c_int, peer_idx as c_int) {
        cudaError_cudaSuccess => Ok(access != 0),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn enable_peer_access(peer_idx: usize) -> CudaResult<()> {
    unsafe {
      match cudaDeviceEnablePeerAccess(peer_idx as c_int, 0) {
        cudaError_cudaSuccess => Ok(()),
        cudaError_cudaErrorPeerAccessAlreadyEnabled => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn disable_peer_access(peer_idx: usize) -> CudaResult<()> {
    unsafe {
      match cudaDeviceDisablePeerAccess(peer_idx as c_int) {
        cudaError_cudaSuccess => Ok(()),
        cudaError_cudaErrorPeerAccessNotEnabled => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

pub struct CudaStream {
  ptr:  cudaStream_t,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaStreamDestroy(self.ptr) {
          cudaError_cudaSuccess => {}
          cudaError_cudaErrorCudartUnloading => {
            // XXX(20160308): Sometimes drop() is called while the global runtime
            // is shutting down; suppress these errors.
          }
          e => panic!("FATAL: CudaStream::drop() failed: {:?} ({})",
              CudaError(e), CudaError(e).get_code()),
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
    unsafe {
      let mut ptr: cudaStream_t = null_mut();
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        cudaError_cudaSuccess => Ok(CudaStream{ptr: ptr}),
        e => Err(CudaError(e)),
      }
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudaStream_t {
    self.ptr
  }

  pub fn add_callback(&mut self, callback: extern "C" fn (stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void), user_data: *mut c_void) -> CudaResult<()> {
    unsafe {
      match cudaStreamAddCallback(self.ptr, Some(callback), user_data, 0) {
        cudaError_cudaSuccess => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn synchronize(&mut self) -> CudaResult<()> {
    unsafe {
      match cudaStreamSynchronize(self.ptr) {
        cudaError_cudaSuccess => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn wait_event(&mut self, event: &mut CudaEvent) -> CudaResult<()> {
    match unsafe { cudaStreamWaitEvent(self.ptr, event.as_mut_ptr(), 0) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e))
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaEventStatus {
  Complete,
  NotReady,
}

pub struct CudaEvent {
  ptr:  cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaEventDestroy(self.ptr) {
          cudaError_cudaSuccess => {}
          cudaError_cudaErrorCudartUnloading => {
            // NB(20160308): Sometimes drop() is called while the global runtime
            // is shutting down; suppress these errors.
          }
          e => panic!("FATAL: CudaEvent::drop(): failed to destroy: {:?}", e),
        }
      }
    }
  }
}

impl CudaEvent {
  pub fn create() -> CudaResult<CudaEvent> {
    unsafe {
      let mut ptr = null_mut() as cudaEvent_t;
      match cudaEventCreate(&mut ptr as *mut cudaEvent_t) {
        cudaError_cudaSuccess => Ok(CudaEvent{ptr: ptr}),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_blocking() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x01)
  }

  pub fn create_fastest() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x02)
  }

  pub fn create_with_flags(flags: u32) -> CudaResult<CudaEvent> {
    unsafe {
      let mut ptr = null_mut() as cudaEvent_t;
      match cudaEventCreateWithFlags(&mut ptr as *mut cudaEvent_t, flags) {
        cudaError_cudaSuccess => Ok(CudaEvent{ptr: ptr}),
        e => Err(CudaError(e)),
      }
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudaEvent_t {
    self.ptr
  }

  pub fn query(&mut self) -> CudaResult<CudaEventStatus> {
    match unsafe { cudaEventQuery(self.ptr) } {
      cudaError_cudaSuccess => Ok(CudaEventStatus::Complete),
      e => match e {
        cudaError_cudaErrorNotReady => Ok(CudaEventStatus::NotReady),
        e => Err(CudaError(e)),
      },
    }
  }

  pub fn record(&mut self, stream: &mut CudaStream) -> CudaResult<()> {
    unsafe {
      match cudaEventRecord(self.ptr, stream.as_mut_ptr()) {
        cudaError_cudaSuccess => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn synchronize(&mut self) -> CudaResult<()> {
    unsafe {
      match cudaEventSynchronize(self.ptr) {
        cudaError_cudaSuccess => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

pub unsafe fn cuda_alloc_host<T>(len: usize) -> CudaResult<*mut T> where T: Copy + 'static {
  let mut ptr: *mut c_void = null_mut();
  let size = len * size_of::<T>();
  match cudaMallocHost(&mut ptr as *mut *mut c_void, size) {
    cudaError_cudaSuccess => Ok(ptr as *mut T),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_alloc_device<T>(len: usize) -> CudaResult<*mut T> where T: Copy + 'static {
  let mut dptr: *mut c_void = null_mut();
  let size = len * size_of::<T>();
  match cudaMalloc(&mut dptr as *mut *mut c_void, size) {
    cudaError_cudaSuccess => Ok(dptr as *mut T),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_device<T>(dptr: *mut T) -> CudaResult<()> where T: Copy + 'static {
  match cudaFree(dptr as *mut c_void) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset(dptr: *mut u8, value: i32, size: usize) -> CudaResult<()> {
  match cudaMemset(dptr as *mut c_void, value, size) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset_async(dptr: *mut u8, value: i32, size: usize, stream: &mut CudaStream) -> CudaResult<()> {
  match cudaMemsetAsync(dptr as *mut c_void, value, size, stream.as_mut_ptr()) {
    cudaError_cudaSuccess => Ok(()),
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
      CudaMemcpyKind::HostToHost      => cudaMemcpyKind_cudaMemcpyHostToHost,
      CudaMemcpyKind::HostToDevice    => cudaMemcpyKind_cudaMemcpyHostToDevice,
      CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind_cudaMemcpyDeviceToHost,
      CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind_cudaMemcpyDeviceToDevice,
      CudaMemcpyKind::Unified         => cudaMemcpyKind_cudaMemcpyDefault,
    }
  }
}

pub unsafe fn cuda_memcpy<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpy(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_async<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind,
    stream: &mut CudaStream) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpyAsync(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw(),
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
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
    stream: &mut CudaStream) -> CudaResult<()>
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
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_peer_async<T>(
    dst: *mut T,
    dst_device_idx: i32,
    src: *const T,
    src_device_idx: i32,
    len: usize,
    stream: &mut CudaStream) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpyPeerAsync(
      dst as *mut c_void,
      dst_device_idx,
      src as *const c_void,
      src_device_idx,
      len * size_of::<T>(),
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}
