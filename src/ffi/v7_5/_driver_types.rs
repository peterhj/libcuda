/* automatically generated by rust-bindgen */

pub const cudaSuccess: cudaError = 0;
pub const cudaErrorMissingConfiguration: cudaError = 1;
pub const cudaErrorMemoryAllocation: cudaError = 2;
pub const cudaErrorInitializationError: cudaError = 3;
pub const cudaErrorLaunchFailure: cudaError = 4;
pub const cudaErrorPriorLaunchFailure: cudaError = 5;
pub const cudaErrorLaunchTimeout: cudaError = 6;
pub const cudaErrorLaunchOutOfResources: cudaError = 7;
pub const cudaErrorInvalidDeviceFunction: cudaError = 8;
pub const cudaErrorInvalidConfiguration: cudaError = 9;
pub const cudaErrorInvalidDevice: cudaError = 10;
pub const cudaErrorInvalidValue: cudaError = 11;
pub const cudaErrorInvalidPitchValue: cudaError = 12;
pub const cudaErrorInvalidSymbol: cudaError = 13;
pub const cudaErrorMapBufferObjectFailed: cudaError = 14;
pub const cudaErrorUnmapBufferObjectFailed: cudaError = 15;
pub const cudaErrorInvalidHostPointer: cudaError = 16;
pub const cudaErrorInvalidDevicePointer: cudaError = 17;
pub const cudaErrorInvalidTexture: cudaError = 18;
pub const cudaErrorInvalidTextureBinding: cudaError = 19;
pub const cudaErrorInvalidChannelDescriptor: cudaError = 20;
pub const cudaErrorInvalidMemcpyDirection: cudaError = 21;
pub const cudaErrorAddressOfConstant: cudaError = 22;
pub const cudaErrorTextureFetchFailed: cudaError = 23;
pub const cudaErrorTextureNotBound: cudaError = 24;
pub const cudaErrorSynchronizationError: cudaError = 25;
pub const cudaErrorInvalidFilterSetting: cudaError = 26;
pub const cudaErrorInvalidNormSetting: cudaError = 27;
pub const cudaErrorMixedDeviceExecution: cudaError = 28;
pub const cudaErrorCudartUnloading: cudaError = 29;
pub const cudaErrorUnknown: cudaError = 30;
pub const cudaErrorNotYetImplemented: cudaError = 31;
pub const cudaErrorMemoryValueTooLarge: cudaError = 32;
pub const cudaErrorInvalidResourceHandle: cudaError = 33;
pub const cudaErrorNotReady: cudaError = 34;
pub const cudaErrorInsufficientDriver: cudaError = 35;
pub const cudaErrorSetOnActiveProcess: cudaError = 36;
pub const cudaErrorInvalidSurface: cudaError = 37;
pub const cudaErrorNoDevice: cudaError = 38;
pub const cudaErrorECCUncorrectable: cudaError = 39;
pub const cudaErrorSharedObjectSymbolNotFound: cudaError = 40;
pub const cudaErrorSharedObjectInitFailed: cudaError = 41;
pub const cudaErrorUnsupportedLimit: cudaError = 42;
pub const cudaErrorDuplicateVariableName: cudaError = 43;
pub const cudaErrorDuplicateTextureName: cudaError = 44;
pub const cudaErrorDuplicateSurfaceName: cudaError = 45;
pub const cudaErrorDevicesUnavailable: cudaError = 46;
pub const cudaErrorInvalidKernelImage: cudaError = 47;
pub const cudaErrorNoKernelImageForDevice: cudaError = 48;
pub const cudaErrorIncompatibleDriverContext: cudaError = 49;
pub const cudaErrorPeerAccessAlreadyEnabled: cudaError = 50;
pub const cudaErrorPeerAccessNotEnabled: cudaError = 51;
pub const cudaErrorDeviceAlreadyInUse: cudaError = 54;
pub const cudaErrorProfilerDisabled: cudaError = 55;
pub const cudaErrorProfilerNotInitialized: cudaError = 56;
pub const cudaErrorProfilerAlreadyStarted: cudaError = 57;
pub const cudaErrorProfilerAlreadyStopped: cudaError = 58;
pub const cudaErrorAssert: cudaError = 59;
pub const cudaErrorTooManyPeers: cudaError = 60;
pub const cudaErrorHostMemoryAlreadyRegistered: cudaError = 61;
pub const cudaErrorHostMemoryNotRegistered: cudaError = 62;
pub const cudaErrorOperatingSystem: cudaError = 63;
pub const cudaErrorPeerAccessUnsupported: cudaError = 64;
pub const cudaErrorLaunchMaxDepthExceeded: cudaError = 65;
pub const cudaErrorLaunchFileScopedTex: cudaError = 66;
pub const cudaErrorLaunchFileScopedSurf: cudaError = 67;
pub const cudaErrorSyncDepthExceeded: cudaError = 68;
pub const cudaErrorLaunchPendingCountExceeded: cudaError = 69;
pub const cudaErrorNotPermitted: cudaError = 70;
pub const cudaErrorNotSupported: cudaError = 71;
pub const cudaErrorHardwareStackError: cudaError = 72;
pub const cudaErrorIllegalInstruction: cudaError = 73;
pub const cudaErrorMisalignedAddress: cudaError = 74;
pub const cudaErrorInvalidAddressSpace: cudaError = 75;
pub const cudaErrorInvalidPc: cudaError = 76;
pub const cudaErrorIllegalAddress: cudaError = 77;
pub const cudaErrorInvalidPtx: cudaError = 78;
pub const cudaErrorInvalidGraphicsContext: cudaError = 79;
pub const cudaErrorStartupFailure: cudaError = 127;
pub const cudaErrorApiFailureBase: cudaError = 10000;
pub type cudaError = u32;
pub const cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyDefault: cudaMemcpyKind = 4;
pub type cudaMemcpyKind = u32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaGraphicsResource {
    _unused: [u8; 0],
}
pub const cudaDevAttrMaxThreadsPerBlock: cudaDeviceAttr = 1;
pub const cudaDevAttrMaxBlockDimX: cudaDeviceAttr = 2;
pub const cudaDevAttrMaxBlockDimY: cudaDeviceAttr = 3;
pub const cudaDevAttrMaxBlockDimZ: cudaDeviceAttr = 4;
pub const cudaDevAttrMaxGridDimX: cudaDeviceAttr = 5;
pub const cudaDevAttrMaxGridDimY: cudaDeviceAttr = 6;
pub const cudaDevAttrMaxGridDimZ: cudaDeviceAttr = 7;
pub const cudaDevAttrMaxSharedMemoryPerBlock: cudaDeviceAttr = 8;
pub const cudaDevAttrTotalConstantMemory: cudaDeviceAttr = 9;
pub const cudaDevAttrWarpSize: cudaDeviceAttr = 10;
pub const cudaDevAttrMaxPitch: cudaDeviceAttr = 11;
pub const cudaDevAttrMaxRegistersPerBlock: cudaDeviceAttr = 12;
pub const cudaDevAttrClockRate: cudaDeviceAttr = 13;
pub const cudaDevAttrTextureAlignment: cudaDeviceAttr = 14;
pub const cudaDevAttrGpuOverlap: cudaDeviceAttr = 15;
pub const cudaDevAttrMultiProcessorCount: cudaDeviceAttr = 16;
pub const cudaDevAttrKernelExecTimeout: cudaDeviceAttr = 17;
pub const cudaDevAttrIntegrated: cudaDeviceAttr = 18;
pub const cudaDevAttrCanMapHostMemory: cudaDeviceAttr = 19;
pub const cudaDevAttrComputeMode: cudaDeviceAttr = 20;
pub const cudaDevAttrMaxTexture1DWidth: cudaDeviceAttr = 21;
pub const cudaDevAttrMaxTexture2DWidth: cudaDeviceAttr = 22;
pub const cudaDevAttrMaxTexture2DHeight: cudaDeviceAttr = 23;
pub const cudaDevAttrMaxTexture3DWidth: cudaDeviceAttr = 24;
pub const cudaDevAttrMaxTexture3DHeight: cudaDeviceAttr = 25;
pub const cudaDevAttrMaxTexture3DDepth: cudaDeviceAttr = 26;
pub const cudaDevAttrMaxTexture2DLayeredWidth: cudaDeviceAttr = 27;
pub const cudaDevAttrMaxTexture2DLayeredHeight: cudaDeviceAttr = 28;
pub const cudaDevAttrMaxTexture2DLayeredLayers: cudaDeviceAttr = 29;
pub const cudaDevAttrSurfaceAlignment: cudaDeviceAttr = 30;
pub const cudaDevAttrConcurrentKernels: cudaDeviceAttr = 31;
pub const cudaDevAttrEccEnabled: cudaDeviceAttr = 32;
pub const cudaDevAttrPciBusId: cudaDeviceAttr = 33;
pub const cudaDevAttrPciDeviceId: cudaDeviceAttr = 34;
pub const cudaDevAttrTccDriver: cudaDeviceAttr = 35;
pub const cudaDevAttrMemoryClockRate: cudaDeviceAttr = 36;
pub const cudaDevAttrGlobalMemoryBusWidth: cudaDeviceAttr = 37;
pub const cudaDevAttrL2CacheSize: cudaDeviceAttr = 38;
pub const cudaDevAttrMaxThreadsPerMultiProcessor: cudaDeviceAttr = 39;
pub const cudaDevAttrAsyncEngineCount: cudaDeviceAttr = 40;
pub const cudaDevAttrUnifiedAddressing: cudaDeviceAttr = 41;
pub const cudaDevAttrMaxTexture1DLayeredWidth: cudaDeviceAttr = 42;
pub const cudaDevAttrMaxTexture1DLayeredLayers: cudaDeviceAttr = 43;
pub const cudaDevAttrMaxTexture2DGatherWidth: cudaDeviceAttr = 45;
pub const cudaDevAttrMaxTexture2DGatherHeight: cudaDeviceAttr = 46;
pub const cudaDevAttrMaxTexture3DWidthAlt: cudaDeviceAttr = 47;
pub const cudaDevAttrMaxTexture3DHeightAlt: cudaDeviceAttr = 48;
pub const cudaDevAttrMaxTexture3DDepthAlt: cudaDeviceAttr = 49;
pub const cudaDevAttrPciDomainId: cudaDeviceAttr = 50;
pub const cudaDevAttrTexturePitchAlignment: cudaDeviceAttr = 51;
pub const cudaDevAttrMaxTextureCubemapWidth: cudaDeviceAttr = 52;
pub const cudaDevAttrMaxTextureCubemapLayeredWidth: cudaDeviceAttr = 53;
pub const cudaDevAttrMaxTextureCubemapLayeredLayers: cudaDeviceAttr = 54;
pub const cudaDevAttrMaxSurface1DWidth: cudaDeviceAttr = 55;
pub const cudaDevAttrMaxSurface2DWidth: cudaDeviceAttr = 56;
pub const cudaDevAttrMaxSurface2DHeight: cudaDeviceAttr = 57;
pub const cudaDevAttrMaxSurface3DWidth: cudaDeviceAttr = 58;
pub const cudaDevAttrMaxSurface3DHeight: cudaDeviceAttr = 59;
pub const cudaDevAttrMaxSurface3DDepth: cudaDeviceAttr = 60;
pub const cudaDevAttrMaxSurface1DLayeredWidth: cudaDeviceAttr = 61;
pub const cudaDevAttrMaxSurface1DLayeredLayers: cudaDeviceAttr = 62;
pub const cudaDevAttrMaxSurface2DLayeredWidth: cudaDeviceAttr = 63;
pub const cudaDevAttrMaxSurface2DLayeredHeight: cudaDeviceAttr = 64;
pub const cudaDevAttrMaxSurface2DLayeredLayers: cudaDeviceAttr = 65;
pub const cudaDevAttrMaxSurfaceCubemapWidth: cudaDeviceAttr = 66;
pub const cudaDevAttrMaxSurfaceCubemapLayeredWidth: cudaDeviceAttr = 67;
pub const cudaDevAttrMaxSurfaceCubemapLayeredLayers: cudaDeviceAttr = 68;
pub const cudaDevAttrMaxTexture1DLinearWidth: cudaDeviceAttr = 69;
pub const cudaDevAttrMaxTexture2DLinearWidth: cudaDeviceAttr = 70;
pub const cudaDevAttrMaxTexture2DLinearHeight: cudaDeviceAttr = 71;
pub const cudaDevAttrMaxTexture2DLinearPitch: cudaDeviceAttr = 72;
pub const cudaDevAttrMaxTexture2DMipmappedWidth: cudaDeviceAttr = 73;
pub const cudaDevAttrMaxTexture2DMipmappedHeight: cudaDeviceAttr = 74;
pub const cudaDevAttrComputeCapabilityMajor: cudaDeviceAttr = 75;
pub const cudaDevAttrComputeCapabilityMinor: cudaDeviceAttr = 76;
pub const cudaDevAttrMaxTexture1DMipmappedWidth: cudaDeviceAttr = 77;
pub const cudaDevAttrStreamPrioritiesSupported: cudaDeviceAttr = 78;
pub const cudaDevAttrGlobalL1CacheSupported: cudaDeviceAttr = 79;
pub const cudaDevAttrLocalL1CacheSupported: cudaDeviceAttr = 80;
pub const cudaDevAttrMaxSharedMemoryPerMultiprocessor: cudaDeviceAttr = 81;
pub const cudaDevAttrMaxRegistersPerMultiprocessor: cudaDeviceAttr = 82;
pub const cudaDevAttrManagedMemory: cudaDeviceAttr = 83;
pub const cudaDevAttrIsMultiGpuBoard: cudaDeviceAttr = 84;
pub const cudaDevAttrMultiGpuBoardGroupID: cudaDeviceAttr = 85;
pub type cudaDeviceAttr = u32;
#[repr(C)]
pub struct cudaDeviceProp {
    pub name: [::std::os::raw::c_char; 256usize],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::std::os::raw::c_int,
    pub warpSize: ::std::os::raw::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::std::os::raw::c_int,
    pub maxThreadsDim: [::std::os::raw::c_int; 3usize],
    pub maxGridSize: [::std::os::raw::c_int; 3usize],
    pub clockRate: ::std::os::raw::c_int,
    pub totalConstMem: usize,
    pub major: ::std::os::raw::c_int,
    pub minor: ::std::os::raw::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::std::os::raw::c_int,
    pub multiProcessorCount: ::std::os::raw::c_int,
    pub kernelExecTimeoutEnabled: ::std::os::raw::c_int,
    pub integrated: ::std::os::raw::c_int,
    pub canMapHostMemory: ::std::os::raw::c_int,
    pub computeMode: ::std::os::raw::c_int,
    pub maxTexture1D: ::std::os::raw::c_int,
    pub maxTexture1DMipmap: ::std::os::raw::c_int,
    pub maxTexture1DLinear: ::std::os::raw::c_int,
    pub maxTexture2D: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DMipmap: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DLinear: [::std::os::raw::c_int; 3usize],
    pub maxTexture2DGather: [::std::os::raw::c_int; 2usize],
    pub maxTexture3D: [::std::os::raw::c_int; 3usize],
    pub maxTexture3DAlt: [::std::os::raw::c_int; 3usize],
    pub maxTextureCubemap: ::std::os::raw::c_int,
    pub maxTexture1DLayered: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DLayered: [::std::os::raw::c_int; 3usize],
    pub maxTextureCubemapLayered: [::std::os::raw::c_int; 2usize],
    pub maxSurface1D: ::std::os::raw::c_int,
    pub maxSurface2D: [::std::os::raw::c_int; 2usize],
    pub maxSurface3D: [::std::os::raw::c_int; 3usize],
    pub maxSurface1DLayered: [::std::os::raw::c_int; 2usize],
    pub maxSurface2DLayered: [::std::os::raw::c_int; 3usize],
    pub maxSurfaceCubemap: ::std::os::raw::c_int,
    pub maxSurfaceCubemapLayered: [::std::os::raw::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::std::os::raw::c_int,
    pub ECCEnabled: ::std::os::raw::c_int,
    pub pciBusID: ::std::os::raw::c_int,
    pub pciDeviceID: ::std::os::raw::c_int,
    pub pciDomainID: ::std::os::raw::c_int,
    pub tccDriver: ::std::os::raw::c_int,
    pub asyncEngineCount: ::std::os::raw::c_int,
    pub unifiedAddressing: ::std::os::raw::c_int,
    pub memoryClockRate: ::std::os::raw::c_int,
    pub memoryBusWidth: ::std::os::raw::c_int,
    pub l2CacheSize: ::std::os::raw::c_int,
    pub maxThreadsPerMultiProcessor: ::std::os::raw::c_int,
    pub streamPrioritiesSupported: ::std::os::raw::c_int,
    pub globalL1CacheSupported: ::std::os::raw::c_int,
    pub localL1CacheSupported: ::std::os::raw::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::std::os::raw::c_int,
    pub managedMemory: ::std::os::raw::c_int,
    pub isMultiGpuBoard: ::std::os::raw::c_int,
    pub multiGpuBoardGroupID: ::std::os::raw::c_int,
}
#[test]
fn bindgen_test_layout_cudaDeviceProp() {
    assert_eq!(
        ::std::mem::size_of::<cudaDeviceProp>(),
        632usize,
        concat!("Size of: ", stringify!(cudaDeviceProp))
    );
    assert_eq!(
        ::std::mem::align_of::<cudaDeviceProp>(),
        8usize,
        concat!("Alignment of ", stringify!(cudaDeviceProp))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).name as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(name)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).totalGlobalMem as *const _ as usize },
        256usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(totalGlobalMem)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).sharedMemPerBlock as *const _ as usize
        },
        264usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(sharedMemPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).regsPerBlock as *const _ as usize },
        272usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(regsPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).warpSize as *const _ as usize },
        276usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(warpSize)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).memPitch as *const _ as usize },
        280usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memPitch)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxThreadsPerBlock as *const _ as usize
        },
        288usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsPerBlock)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxThreadsDim as *const _ as usize },
        292usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsDim)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxGridSize as *const _ as usize },
        304usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxGridSize)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).clockRate as *const _ as usize },
        316usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(clockRate)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).totalConstMem as *const _ as usize },
        320usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(totalConstMem)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).major as *const _ as usize },
        328usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(major)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).minor as *const _ as usize },
        332usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(minor)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).textureAlignment as *const _ as usize },
        336usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(textureAlignment)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).texturePitchAlignment as *const _ as usize
        },
        344usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(texturePitchAlignment)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).deviceOverlap as *const _ as usize },
        352usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(deviceOverlap)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).multiProcessorCount as *const _ as usize
        },
        356usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(multiProcessorCount)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).kernelExecTimeoutEnabled as *const _ as usize
        },
        360usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(kernelExecTimeoutEnabled)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).integrated as *const _ as usize },
        364usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(integrated)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).canMapHostMemory as *const _ as usize },
        368usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(canMapHostMemory)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).computeMode as *const _ as usize },
        372usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(computeMode)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture1D as *const _ as usize },
        376usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1D)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture1DMipmap as *const _ as usize
        },
        380usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DMipmap)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture1DLinear as *const _ as usize
        },
        384usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DLinear)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture2D as *const _ as usize },
        388usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2D)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture2DMipmap as *const _ as usize
        },
        396usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DMipmap)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture2DLinear as *const _ as usize
        },
        404usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DLinear)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture2DGather as *const _ as usize
        },
        416usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DGather)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture3D as *const _ as usize },
        424usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture3D)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture3DAlt as *const _ as usize },
        436usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture3DAlt)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTextureCubemap as *const _ as usize
        },
        448usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTextureCubemap)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture1DLayered as *const _ as usize
        },
        452usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture1DLayered)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTexture2DLayered as *const _ as usize
        },
        460usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTexture2DLayered)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxTextureCubemapLayered as *const _ as usize
        },
        472usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxTextureCubemapLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurface1D as *const _ as usize },
        480usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface1D)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurface2D as *const _ as usize },
        484usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface2D)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurface3D as *const _ as usize },
        492usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface3D)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurface1DLayered as *const _ as usize
        },
        504usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface1DLayered)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurface2DLayered as *const _ as usize
        },
        512usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurface2DLayered)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurfaceCubemap as *const _ as usize
        },
        524usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurfaceCubemap)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxSurfaceCubemapLayered as *const _ as usize
        },
        528usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxSurfaceCubemapLayered)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).surfaceAlignment as *const _ as usize },
        536usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(surfaceAlignment)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).concurrentKernels as *const _ as usize
        },
        544usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(concurrentKernels)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).ECCEnabled as *const _ as usize },
        548usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(ECCEnabled)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).pciBusID as *const _ as usize },
        552usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciBusID)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).pciDeviceID as *const _ as usize },
        556usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciDeviceID)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).pciDomainID as *const _ as usize },
        560usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(pciDomainID)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).tccDriver as *const _ as usize },
        564usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(tccDriver)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).asyncEngineCount as *const _ as usize },
        568usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(asyncEngineCount)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).unifiedAddressing as *const _ as usize
        },
        572usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(unifiedAddressing)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).memoryClockRate as *const _ as usize },
        576usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memoryClockRate)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).memoryBusWidth as *const _ as usize },
        580usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(memoryBusWidth)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).l2CacheSize as *const _ as usize },
        584usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(l2CacheSize)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).maxThreadsPerMultiProcessor as *const _
                as usize
        },
        588usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(maxThreadsPerMultiProcessor)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).streamPrioritiesSupported as *const _
                as usize
        },
        592usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(streamPrioritiesSupported)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).globalL1CacheSupported as *const _ as usize
        },
        596usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(globalL1CacheSupported)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).localL1CacheSupported as *const _ as usize
        },
        600usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(localL1CacheSupported)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).sharedMemPerMultiprocessor as *const _
                as usize
        },
        608usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(sharedMemPerMultiprocessor)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).regsPerMultiprocessor as *const _ as usize
        },
        616usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(regsPerMultiprocessor)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).managedMemory as *const _ as usize },
        620usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(managedMemory)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<cudaDeviceProp>())).isMultiGpuBoard as *const _ as usize },
        624usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(isMultiGpuBoard)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<cudaDeviceProp>())).multiGpuBoardGroupID as *const _ as usize
        },
        628usize,
        concat!(
            "Offset of field: ",
            stringify!(cudaDeviceProp),
            "::",
            stringify!(multiGpuBoardGroupID)
        )
    );
}
pub use self::cudaError as cudaError_t;
pub type cudaStream_t = *mut CUstream_st;
pub type cudaEvent_t = *mut CUevent_st;
pub type cudaGraphicsResource_t = *mut cudaGraphicsResource;
pub type cudaUUID_t = CUuuid_st;
