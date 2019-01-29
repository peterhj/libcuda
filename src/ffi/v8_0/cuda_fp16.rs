// NB: clang does not seem to like cuda_fp16.h in CUDA 8.0.
// Instead of deferring to bindgen, write and test manual bindings.

#[repr(C)]
pub struct __half {
  pub x: ::std::os::raw::c_ushort,
}

#[test]
fn nonbindgen_test_layout___half() {
  assert_eq!(::std::mem::size_of::<__half>(), 2usize,
      concat!("Size of: ", stringify!(__half)));
  assert_eq!(::std::mem::align_of::<__half>(), 2usize,
      concat!("Align of: ", stringify!(__half)));
}

#[repr(C)]
pub struct __half2 {
  pub x: ::std::os::raw::c_uint,
}

#[test]
fn nonbindgen_test_layout___half2() {
  assert_eq!(::std::mem::size_of::<__half2>(), 4usize,
      concat!("Size of: ", stringify!(__half2)));
  assert_eq!(::std::mem::align_of::<__half2>(), 4usize,
      concat!("Align of: ", stringify!(__half2)));
}
