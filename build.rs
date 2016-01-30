#![feature(stmt_expr_attributes)]

fn main() {
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cudart");
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l static=cudadevrt -l dylib=cudart");

  //println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cudart");
  println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cudart");

  //println!("cargo:rustc-link-lib=native=cudart");
  //println!("cargo:rustc-link-search=native=/usr/local/cuda-7.0/lib64");

  /*#[cfg(feature = "cuda-7-0")]
  {
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-search=dylib=/usr/local/cuda-7.0/lib64");
  }

  #[cfg(feature = "cuda-7-5")]
  {
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-search=dylib=/usr/local/cuda-7.5/lib64");
  }*/
}
