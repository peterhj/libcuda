#![allow(missing_copy_implementations)]

use ffi::runtime::*;

use libc::{c_void, c_int, c_uint, size_t};
use std::mem::{size_of, transmute};
use std::ops::{Range};
use std::ptr::{null_mut};

#[repr(C)]
pub struct Dim3 {
  x: u32,
  y: u32,
  z: u32,
}

pub type CudaResult<T> = Result<T, CudaError>;

#[derive(Clone, Copy, Debug)]
pub struct CudaError(cudaError_t);

impl CudaError {
  /*pub fn get_name(&self) -> &mut str {
    let &CudaError(e) = self;
    unsafe {
      from_raw_mut_buf(cudaGetErrorName(e));
    }
  }

  pub fn get_string(&self) -> &mut str {
  }*/

  pub fn get_code(&self) -> i64 {
    let &CudaError(ref e) = self;
    unsafe {
      transmute(e)
    }
  }
}

pub fn cuda_get_driver_version() -> CudaResult<i32> {
  unsafe {
    let mut version: c_int = 0;
    match cudaDriverGetVersion(&mut version as *mut c_int) {
      cudaError_t::Success => Ok(version as i32),
      e => Err(CudaError(e)),
    }
  }
}

pub fn cuda_get_runtime_version() -> CudaResult<i32> {
  unsafe {
    let mut version: c_int = 0;
    match cudaRuntimeGetVersion(&mut version as *mut c_int) {
      cudaError_t::Success => Ok(version as i32),
      e => Err(CudaError(e)),
    }
  }
}

// TODO: device flags.

pub struct CudaDevice;

impl CudaDevice {
  pub fn count() -> CudaResult<usize> {
    let mut count: c_int = 0;
    unsafe {
      match cudaGetDeviceCount(&mut count as *mut c_int) {
        cudaError_t::Success => Ok(count as usize),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn iter() -> CudaResult<Range<usize>> {
    Self::count().and_then(|count| Ok((0 .. count)))
  }

  pub fn get_properties(&self) {
    /*unsafe {
      match cudaGetProperties(...) {
      }
    }*/
  }

  pub fn get_current() -> CudaResult<usize> {
    let mut index: c_int = 0;
    match unsafe { cudaGetDevice(&mut index as *mut c_int) } {
      cudaError_t::Success => Ok(index as usize),
      e => Err(CudaError(e)),
    }
  }

  pub fn set_current(index: usize) -> CudaResult<()> {
    unsafe {
      match cudaSetDevice(index as c_int) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn reset() -> CudaResult<()> {
    match unsafe { cudaDeviceReset() } {
      cudaError_t::Success => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn set_flags(flags: u32) -> CudaResult<()> {
    match unsafe { cudaSetDeviceFlags(flags as c_uint) } {
      cudaError_t::Success => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn get_attribute(device_idx: usize, ffi_attr: cudaDeviceAttr) -> CudaResult<i32> {
    let mut value: c_int = 0;
    match unsafe { cudaDeviceGetAttribute(&mut value as *mut c_int, ffi_attr, device_idx as c_int) } {
      cudaError_t::Success => Ok(value as i32),
      e => Err(CudaError(e)),
    }
  }

  pub fn can_access_peer(idx: usize, peer_idx: usize) -> CudaResult<bool> {
    unsafe {
      let mut access: c_int = 0;
      match cudaDeviceCanAccessPeer(&mut access as *mut c_int, idx as c_int, peer_idx as c_int) {
        cudaError_t::Success => Ok(access != 0),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn enable_peer_access(peer_idx: usize) -> CudaResult<()> {
    unsafe {
      match cudaDeviceEnablePeerAccess(peer_idx as c_int, 0) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn disable_peer_access(peer_idx: usize) -> CudaResult<()> {
    unsafe {
      match cudaDeviceDisablePeerAccess(peer_idx as c_int) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

pub struct CudaStream {
  pub ptr: cudaStream_t,
}

//impl !Send for CudaStream {}
//impl !Sync for CudaStream {}
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaStreamDestroy(self.ptr) {
          cudaError_t::Success => {}
          cudaError_t::CudartUnloading => {
            // XXX(20160308): Sometimes drop() is called while the global runtime
            // is shutting down; suppress these errors.
          }
          e => panic!("FATAL: CudaStream::drop() failed: {}", CudaError(e).get_code()),
        }
      }
    }
  }
}

impl CudaStream {
  pub fn default() -> CudaStream {
    CudaStream{
      ptr: null_mut(),
    }
  }

  pub fn create() -> CudaResult<CudaStream> {
    unsafe {
      let mut ptr: cudaStream_t = null_mut();
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        cudaError_t::Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_with_flags(_flags: i32) -> CudaResult<CudaStream> {
    unimplemented!();
    /*unsafe {
      // TODO: flags.
      let mut ptr: cudaStream_t = null_mut();
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        cudaError_t::Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }*/
  }

  pub fn create_with_priority(_flags: i32, _priority: i32) -> CudaResult<CudaStream> {
    unimplemented!();
    /*unsafe {
      // TODO: flags and priority.
      let mut ptr: cudaStream_t = null_mut();
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        cudaError_t::Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }*/
  }

  pub fn add_callback(&self, callback: extern "C" fn (stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void), user_data: *mut c_void) -> CudaResult<()> {
    unsafe {
      match cudaStreamAddCallback(self.ptr, callback, user_data, 0) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn synchronize(&self) -> CudaResult<()> {
    unsafe {
      match cudaStreamSynchronize(self.ptr) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
    match unsafe { cudaStreamWaitEvent(self.ptr, event.ptr, 0) } {
      cudaError_t::Success => Ok(()),
      e => Err(CudaError(e))
    }
  }

  pub unsafe fn as_ptr(&self) -> cudaStream_t {
    self.ptr
  }
}

pub enum CudaEventStatus {
  Complete,
  NotReady,
}

pub struct CudaEvent {
  pub ptr: cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaEventDestroy(self.ptr) {
          cudaError_t::Success => {}
          cudaError_t::CudartUnloading => {
            // XXX(20160308): Sometimes drop() is called while the global runtime
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
      let mut ptr = 0 as cudaEvent_t;
      match cudaEventCreate(&mut ptr as *mut cudaEvent_t) {
        cudaError_t::Success => {
          Ok(CudaEvent{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_fastest() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x02)
  }

  pub fn create_with_flags(flags: u32) -> CudaResult<CudaEvent> {
    unsafe {
      let mut ptr = 0 as cudaEvent_t;
      match cudaEventCreateWithFlags(&mut ptr as *mut cudaEvent_t, flags as c_uint) {
        cudaError_t::Success => {
          Ok(CudaEvent{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn record(&self, stream: &CudaStream) -> CudaResult<()> {
    unsafe {
      match cudaEventRecord(self.ptr, stream.ptr) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn query(&self) -> CudaResult<CudaEventStatus> {
    match unsafe { cudaEventQuery(self.ptr) } {
      cudaError_t::Success => Ok(CudaEventStatus::Complete),
      e => match e {
        cudaError_t::NotReady => Ok(CudaEventStatus::NotReady),
        e => Err(CudaError(e)),
      },
    }
  }

  pub fn synchronize(&self) -> CudaResult<()> {
    unsafe {
      match cudaEventSynchronize(self.ptr) {
        cudaError_t::Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaMemInfo {
  pub used: usize,
  pub free: usize,
  pub total: usize,
}

pub fn cuda_get_mem_info() -> CudaResult<CudaMemInfo> {
  unsafe {
    let mut free: size_t = 0;
    let mut total: size_t = 0;
    match cudaMemGetInfo(&mut free as *mut size_t, &mut total as *mut size_t) {
      cudaError_t::Success => Ok(CudaMemInfo{
        used: (total - free) as usize,
        free: free as usize,
        total: total as usize,
      }),
      e => Err(CudaError(e)),
    }
  }
}

pub unsafe fn cuda_alloc_pinned(size: usize, flags: u32) -> CudaResult<*mut u8> {
  let mut ptr = 0 as *mut c_void;
  match cudaHostAlloc(&mut ptr as *mut *mut c_void, size as size_t, flags) {
    cudaError_t::Success => Ok(ptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_pinned(ptr: *mut u8) -> CudaResult<()> {
  match cudaFreeHost(ptr as *mut c_void) {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_alloc_device<T>(len: usize) -> CudaResult<*mut T> where T: Copy {
  let mut ptr: *mut c_void = null_mut();
  let size = len * size_of::<T>();
  match cudaMalloc(&mut ptr as *mut *mut c_void, size) {
    cudaError_t::Success => Ok(ptr as *mut T),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_device<T>(dev_ptr: *mut T) -> CudaResult<()> where T: Copy {
  match cudaFree(dev_ptr as *mut c_void) {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset(dev_ptr: *mut u8, value: i32, size: usize) -> CudaResult<()> {
  match cudaMemset(dev_ptr as *mut c_void, value, size as size_t) {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset_async(dev_ptr: *mut u8, value: i32, size: usize, stream: &CudaStream) -> CudaResult<()> {
  match cudaMemsetAsync(dev_ptr as *mut c_void, value, size as size_t, stream.ptr) {
    cudaError_t::Success => Ok(()),
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
  pub fn to_ffi(&self) -> cudaMemcpyKind {
    match *self {
      CudaMemcpyKind::HostToHost      => cudaMemcpyKind::HostToHost,
      CudaMemcpyKind::HostToDevice    => cudaMemcpyKind::HostToDevice,
      CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind::DeviceToHost,
      CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind::DeviceToDevice,
      CudaMemcpyKind::Unified         => cudaMemcpyKind::Default,
    }
  }
}

pub unsafe fn cuda_memcpy<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind) -> CudaResult<()>
where T: Copy
{
  match cudaMemcpy(
      dst as *mut c_void,
      src as *const c_void,
      (len * size_of::<T>()) as size_t,
      kind.to_ffi())
  {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_async<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind,
    stream: &CudaStream) -> CudaResult<()>
where T: Copy
{
  match cudaMemcpyAsync(
      dst as *mut c_void,
      src as *const c_void,
      (len * size_of::<T>()) as size_t,
      kind.to_ffi(),
      stream.ptr)
  {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_peer_async<T>(
    dst: *mut T, dst_device_idx: usize,
    src: *const T, src_device_idx: usize,
    len: usize,
    stream: &CudaStream) -> CudaResult<()>
where T: Copy
{
  match cudaMemcpyPeerAsync(
      dst as *mut c_void, dst_device_idx as c_int,
      src as *const c_void, src_device_idx as c_int,
      (len * size_of::<T>()) as size_t,
      stream.ptr)
  {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_2d(
    dst: *mut u8, dst_pitch: usize,
    src: *const u8, src_pitch: usize,
    width: usize, height: usize,
    kind: CudaMemcpyKind) -> CudaResult<()>
{
  match cudaMemcpy2D(
      dst as *mut c_void, dst_pitch as size_t,
      src as *const c_void, src_pitch as size_t,
      width as size_t, height as size_t,
      kind.to_ffi())
  {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_2d_async(
    dst: *mut u8, dst_pitch: usize,
    src: *const u8, src_pitch: usize,
    width: usize, height: usize,
    kind: CudaMemcpyKind,
    stream: &CudaStream) -> CudaResult<()>
{
  match cudaMemcpy2DAsync(
      dst as *mut c_void, dst_pitch as size_t,
      src as *const c_void, src_pitch as size_t,
      width as size_t, height as size_t,
      kind.to_ffi(),
      stream.ptr)
  {
    cudaError_t::Success => Ok(()),
    e => Err(CudaError(e)),
  }
}
