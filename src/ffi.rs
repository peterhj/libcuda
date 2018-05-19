#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod library_types {
include!(concat!(env!("OUT_DIR"), "/libtypes_bind.rs"));
}

pub mod runtime {
include!(concat!(env!("OUT_DIR"), "/runtime_bind.rs"));
}
