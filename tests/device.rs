extern crate cuda;

use cuda::runtime::{CudaDevice};

#[test]
fn device_count() {
  let maybe_dev_ct = CudaDevice::count();
  println!("DEBUG: device count? {:?}", maybe_dev_ct);
  match maybe_dev_ct {
    Err(_) => panic!(),
    Ok(ct) => assert!(ct >= 1),
  }
}
