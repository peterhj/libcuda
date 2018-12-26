#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_ffi_types::cuda::*;
include!(concat!(env!("OUT_DIR"), "/cuda.rs"));
