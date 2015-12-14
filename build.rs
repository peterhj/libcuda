fn main() {
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cudart");
  println!("cargo:rustc-flags=-l dylib=cudart");
}
