use runtime::*;

#[test]
fn devices() {
  let count = Device::count().ok().unwrap();
  assert!(count >= 1);
  let device = Device::new(0);
  device.set_current().ok();
}

#[test]
fn pinned_memory() {
  let ptr = cuda_host_alloc_pinned(4096, 0).ok().unwrap();
  assert!(ptr as usize != 0);
  cuda_host_free_pinned(ptr).ok();
}
