extern crate cuda;

use cuda::driver::*;

#[test]
fn test_init() {
  assert!(!is_cuda_initialized());
  cuda_init();
  assert!(is_cuda_initialized());
}

#[test]
fn test_version_no_init() {
  let v = get_version();
  println!("version? {:?}", v);
  assert!(v.is_ok());
}
