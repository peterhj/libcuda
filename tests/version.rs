extern crate cuda;

use cuda::runtime::*;

#[test]
fn test_version_driver() {
  println!("driver version? {:?}", get_driver_version());
}

#[test]
fn test_version_runtime() {
  println!("runtime version? {:?}", get_runtime_version());
}
