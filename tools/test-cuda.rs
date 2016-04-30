extern crate cuda;

use cuda::runtime::*;

fn main() {
  println!("{}", CudaDevice::count().unwrap());
}
