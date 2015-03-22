fn main() {
  println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l cudart:dylib -l cublas:dylib -l cufft:dylib");
}
