extern crate cuda;

use cuda::driver::*;

#[test]
fn test_init() {
  assert!(!cuda_initialized().unwrap_or_else(|_| false));
  cuda_init().ok();
  assert!(cuda_initialized().unwrap_or_else(|_| false));
}

#[test]
fn test_version_no_init() {
  let v = get_version();
  println!("version? {:?}", v);
  assert!(v.is_ok());
}
