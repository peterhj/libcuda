/* automatically generated by rust-bindgen */

pub const CURAND_STATUS_SUCCESS: curandStatus = 0;
pub const CURAND_STATUS_VERSION_MISMATCH: curandStatus = 100;
pub const CURAND_STATUS_NOT_INITIALIZED: curandStatus = 101;
pub const CURAND_STATUS_ALLOCATION_FAILED: curandStatus = 102;
pub const CURAND_STATUS_TYPE_ERROR: curandStatus = 103;
pub const CURAND_STATUS_OUT_OF_RANGE: curandStatus = 104;
pub const CURAND_STATUS_LENGTH_NOT_MULTIPLE: curandStatus = 105;
pub const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: curandStatus = 106;
pub const CURAND_STATUS_LAUNCH_FAILURE: curandStatus = 201;
pub const CURAND_STATUS_PREEXISTING_FAILURE: curandStatus = 202;
pub const CURAND_STATUS_INITIALIZATION_FAILED: curandStatus = 203;
pub const CURAND_STATUS_ARCH_MISMATCH: curandStatus = 204;
pub const CURAND_STATUS_INTERNAL_ERROR: curandStatus = 999;
pub type curandStatus = u32;
pub use self::curandStatus as curandStatus_t;
pub const CURAND_RNG_TEST: curandRngType = 0;
pub const CURAND_RNG_PSEUDO_DEFAULT: curandRngType = 100;
pub const CURAND_RNG_PSEUDO_XORWOW: curandRngType = 101;
pub const CURAND_RNG_PSEUDO_MRG32K3A: curandRngType = 121;
pub const CURAND_RNG_PSEUDO_MTGP32: curandRngType = 141;
pub const CURAND_RNG_PSEUDO_MT19937: curandRngType = 142;
pub const CURAND_RNG_PSEUDO_PHILOX4_32_10: curandRngType = 161;
pub const CURAND_RNG_QUASI_DEFAULT: curandRngType = 200;
pub const CURAND_RNG_QUASI_SOBOL32: curandRngType = 201;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: curandRngType = 202;
pub const CURAND_RNG_QUASI_SOBOL64: curandRngType = 203;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: curandRngType = 204;
pub type curandRngType = u32;
pub use self::curandRngType as curandRngType_t;
pub const CURAND_ORDERING_PSEUDO_BEST: curandOrdering = 100;
pub const CURAND_ORDERING_PSEUDO_DEFAULT: curandOrdering = 101;
pub const CURAND_ORDERING_PSEUDO_SEEDED: curandOrdering = 102;
pub const CURAND_ORDERING_QUASI_DEFAULT: curandOrdering = 201;
pub type curandOrdering = u32;
pub use self::curandOrdering as curandOrdering_t;
pub const CURAND_DIRECTION_VECTORS_32_JOEKUO6: curandDirectionVectorSet = 101;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6: curandDirectionVectorSet = 102;
pub const CURAND_DIRECTION_VECTORS_64_JOEKUO6: curandDirectionVectorSet = 103;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6: curandDirectionVectorSet = 104;
pub type curandDirectionVectorSet = u32;
pub use self::curandDirectionVectorSet as curandDirectionVectorSet_t;
pub type curandDirectionVectors32_t = [::std::os::raw::c_uint; 32usize];
pub type curandDirectionVectors64_t = [::std::os::raw::c_ulonglong; 64usize];
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandGenerator_st {
    _unused: [u8; 0],
}
pub type curandGenerator_t = *mut curandGenerator_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDiscreteDistribution_st {
    _unused: [u8; 0],
}
pub type curandDiscreteDistribution_t = *mut curandDiscreteDistribution_st;
extern "C" {
    pub fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandCreateGeneratorHost(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetVersion(version: *mut ::std::os::raw::c_int) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetProperty(
        type_: libraryPropertyType,
        value: *mut ::std::os::raw::c_int,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: ::std::os::raw::c_ulonglong,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetGeneratorOffset(
        generator: curandGenerator_t,
        offset: ::std::os::raw::c_ulonglong,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetGeneratorOrdering(
        generator: curandGenerator_t,
        order: curandOrdering_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandSetQuasiRandomGeneratorDimensions(
        generator: curandGenerator_t,
        num_dimensions: ::std::os::raw::c_uint,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerate(
        generator: curandGenerator_t,
        outputPtr: *mut ::std::os::raw::c_uint,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLongLong(
        generator: curandGenerator_t,
        outputPtr: *mut ::std::os::raw::c_ulonglong,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateUniform(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        num: usize,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLogNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateLogNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandCreatePoissonDistribution(
        lambda: f64,
        discrete_distribution: *mut curandDiscreteDistribution_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandDestroyDistribution(
        discrete_distribution: curandDiscreteDistribution_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGeneratePoisson(
        generator: curandGenerator_t,
        outputPtr: *mut ::std::os::raw::c_uint,
        n: usize,
        lambda: f64,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetDirectionVectors32(
        vectors: *mut *mut curandDirectionVectors32_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetScrambleConstants32(
        constants: *mut *mut ::std::os::raw::c_uint,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetDirectionVectors64(
        vectors: *mut *mut curandDirectionVectors64_t,
        set: curandDirectionVectorSet_t,
    ) -> curandStatus_t;
}
extern "C" {
    pub fn curandGetScrambleConstants64(
        constants: *mut *mut ::std::os::raw::c_ulonglong,
    ) -> curandStatus_t;
}
