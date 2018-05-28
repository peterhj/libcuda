extern crate cuda;

use cuda::driver::*;

#[test]
fn test_cuda() {
  println!();
  println!("DEBUG: test cuda init...");
  println!("DEBUG: cuda init? {:?}", is_cuda_initialized());
  println!("DEBUG: cuda init...");
  cuda_init();
  println!("DEBUG: test cuda init...");
  println!("DEBUG: cuda init? {:?}", is_cuda_initialized());
}
