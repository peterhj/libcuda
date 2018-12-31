#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub use self::v::cuda::*;

#[cfg(feature = "cuda_8_0")]
mod v {
  pub mod cuda {
    use cuda_api_types::cuda::*;
    include!("v8_0/_cuda.rs");
  }
}

#[cfg(feature = "cuda_9_0")]
mod v {
  pub mod cuda {
    use cuda_api_types::cuda::*;
    include!("v9_0/_cuda.rs");
  }
}

#[cfg(feature = "cuda_9_2")]
mod v {
  pub mod cuda {
    use cuda_api_types::cuda::*;
    include!("v9_2/_cuda.rs");
  }
}
