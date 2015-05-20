#![allow(missing_copy_implementations)]

use ffi::runtime::*;
use ffi::runtime::cudaError::{Success};

use libc::{c_void, c_int, c_uint, size_t};
//use libc::funcs::c95::{strlen};
use std::mem::{transmute};

#[repr(C)]
pub struct Dim3 {
  x: u32,
  y: u32,
  z: u32,
}

pub type CudaResult<T> = Result<T, CudaError>;

#[derive(Clone, Copy, Debug)]
pub struct CudaError(cudaError);

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
      Success => Ok(version as i32),
      e => Err(CudaError(e)),
    }
  }
}

pub fn cuda_get_runtime_version() -> CudaResult<i32> {
  unsafe {
    let mut version: c_int = 0;
    match cudaRuntimeGetVersion(&mut version as *mut c_int) {
      Success => Ok(version as i32),
      e => Err(CudaError(e)),
    }
  }
}

// TODO: device flags.

pub struct Device;

impl Device {
  pub fn count() -> CudaResult<usize> {
    let mut count: c_int = 0;
    unsafe {
      match cudaGetDeviceCount(&mut count as *mut c_int) {
        Success => Ok(count as usize),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn get_properties(&self) {
    /*unsafe {
      match cudaGetProperties(...) {
      }
    }*/
  }

  pub fn set_current(index: usize) -> CudaResult<()> {
    unsafe {
      match cudaSetDevice(index as c_int) {
        Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn reset_with_flags(flags: u32) -> CudaResult<()> {
    unsafe {
      cudaDeviceReset();
      cudaSetDeviceFlags(flags);
    }
    Ok(())
  }
}

pub struct CudaStream {
  pub ptr: cudaStream_t,
}

impl !Send for CudaStream {
}

impl Drop for CudaStream {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaStreamDestroy(self.ptr) {
          Success => (),
          e => panic!("FATAL: CudaStream::drop() failed: {}", CudaError(e).get_code()),
        }
      }
    }
  }
}

impl CudaStream {
  pub fn default() -> CudaStream {
    CudaStream{
      ptr: 0 as cudaStream_t,
    }
  }

  pub fn create() -> CudaResult<CudaStream> {
    unsafe {
      let mut ptr = 0 as cudaStream_t;
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_with_flags(flags: i32) -> CudaResult<CudaStream> {
    unsafe {
      // TODO: flags.
      let mut ptr = 0 as cudaStream_t;
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_with_priority(flags: i32, priority: i32) -> CudaResult<CudaStream> {
    unsafe {
      // TODO: flags and priority.
      let mut ptr = 0 as cudaStream_t;
      match cudaStreamCreate(&mut ptr as *mut cudaStream_t) {
        Success => {
          Ok(CudaStream{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn synchronize(&self) -> CudaResult<()> {
    unsafe {
      match cudaStreamSynchronize(self.ptr) {
        Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

pub struct CudaEvent {
  pub ptr: cudaEvent_t,
}

impl !Send for CudaEvent {
}

impl Drop for CudaEvent {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      unsafe {
        match cudaEventDestroy(self.ptr) {
          Success => (),
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
        Success => {
          Ok(CudaEvent{
            ptr: ptr,
          })
        },
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn create_with_flags(flags: u32) -> CudaResult<CudaEvent> {
    unsafe {
      let mut ptr = 0 as cudaEvent_t;
      match cudaEventCreateWithFlags(&mut ptr as *mut cudaEvent_t, flags as c_uint) {
        Success => {
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
        Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }

  pub fn synchronize(&self) -> CudaResult<()> {
    unsafe {
      match cudaEventSynchronize(self.ptr) {
        Success => Ok(()),
        e => Err(CudaError(e)),
      }
    }
  }
}

pub struct CudaMemInfo {
  pub free: usize,
  pub total: usize,
}

pub fn cuda_get_mem_info() -> CudaResult<CudaMemInfo> {
  unsafe {
    let mut free: size_t = 0;
    let mut total: size_t = 0;
    match cudaMemGetInfo(&mut free as *mut size_t, &mut total as *mut size_t) {
      Success => Ok(CudaMemInfo{
        free: free as usize,
        total: total as usize,
      }),
      e => Err(CudaError(e)),
    }
  }
}

pub fn cuda_alloc_pinned(size: usize, flags: u32) -> CudaResult<*mut u8> {
  unsafe {
    let mut ptr = 0 as *mut c_void;
    match cudaHostAlloc(&mut ptr as *mut *mut c_void, size as size_t, flags) {
      Success => Ok(ptr as *mut u8),
      e => Err(CudaError(e)),
    }
  }
}

pub unsafe fn cuda_free_pinned(ptr: *mut u8) -> CudaResult<()> {
  match cudaFreeHost(ptr as *mut c_void) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub fn cuda_alloc_device(size: usize) -> CudaResult<*mut u8> {
  //println!("DEBUG: calling alloc_device()");
  unsafe {
    let mut ptr = 0 as *mut c_void;
    match cudaMalloc(&mut ptr as *mut *mut c_void, size as u64) {
      Success => Ok(ptr as *mut u8),
      e => Err(CudaError(e)),
    }
  }
}

pub unsafe fn cuda_free_device(dev_ptr: *mut u8) -> CudaResult<()> {
  match cudaFree(dev_ptr as *mut c_void) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset(dev_ptr: *mut u8, value: i32, size: usize) -> CudaResult<()> {
  match cudaMemset(dev_ptr as *mut c_void, value, size as size_t) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset_async(dev_ptr: *mut u8, value: i32, size: usize, stream: &CudaStream) -> CudaResult<()> {
  match cudaMemsetAsync(dev_ptr as *mut c_void, value, size as size_t, stream.ptr) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub enum CudaMemcpyKind {
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  Unified,
}

pub unsafe fn cuda_memcpy(dst: *mut u8, src: *const u8, size: usize, kind: CudaMemcpyKind) -> CudaResult<()> {
  let kind = match kind {
    CudaMemcpyKind::HostToHost      => cudaMemcpyKind::HostToHost,
    CudaMemcpyKind::HostToDevice    => cudaMemcpyKind::HostToDevice,
    CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind::DeviceToHost,
    CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind::DeviceToDevice,
    CudaMemcpyKind::Unified         => cudaMemcpyKind::Default,
  };
  match cudaMemcpy(dst as *mut c_void, src as *const c_void, size as size_t, kind) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_async(dst: *mut u8, src: *const u8, size: usize, kind: CudaMemcpyKind, stream: &CudaStream) -> CudaResult<()> {
  let kind = match kind {
    CudaMemcpyKind::HostToHost      => cudaMemcpyKind::HostToHost,
    CudaMemcpyKind::HostToDevice    => cudaMemcpyKind::HostToDevice,
    CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind::DeviceToHost,
    CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind::DeviceToDevice,
    CudaMemcpyKind::Unified         => cudaMemcpyKind::Default,
  };
  match cudaMemcpyAsync(dst as *mut c_void, src as *const c_void, size as size_t, kind, stream.ptr) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_2d(dst: *mut u8, dst_pitch: usize, src: *const u8, src_pitch: usize, width: usize, height: usize, kind: CudaMemcpyKind) -> CudaResult<()> {
  let kind = match kind {
    CudaMemcpyKind::HostToHost      => cudaMemcpyKind::HostToHost,
    CudaMemcpyKind::HostToDevice    => cudaMemcpyKind::HostToDevice,
    CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind::DeviceToHost,
    CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind::DeviceToDevice,
    CudaMemcpyKind::Unified         => cudaMemcpyKind::Default,
  };
  match cudaMemcpy2D(dst as *mut c_void, dst_pitch as size_t, src as *const c_void, src_pitch as size_t, width as size_t, height as size_t, kind) {
    Success => Ok(()),
    e => Err(CudaError(e)),
  }
}
