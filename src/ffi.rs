#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod driver {
include!(concat!(env!("OUT_DIR"), "/driver_bind.rs"));
}

pub mod driver_types {
use ffi::driver::*;
include!(concat!(env!("OUT_DIR"), "/driver_types_bind.rs"));
}

pub mod library_types {
include!(concat!(env!("OUT_DIR"), "/libtypes_bind.rs"));
}

pub mod runtime {
use ffi::driver_types::*;
include!(concat!(env!("OUT_DIR"), "/runtime_bind.rs"));
}
