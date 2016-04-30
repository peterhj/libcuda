extern crate cuda;

use cuda::runtime::*;

fn main() {
  println!("DEBUG: cuda device count: {}", CudaDevice::count().unwrap());
}
