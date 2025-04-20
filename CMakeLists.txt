cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

option(ENABLE_NCNN_BACKEND "Enable NCNN backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)
option(ENABLE_LLAMACPP_BACKEND "Enable llama.cpp backend" ON)
option(ENABLE_QNN_BACKEND "Enable QNN backend" OFF)

option(ENABLE_VISION "Enable vision" OFF)
option(ENABLE_WHISPER "Enable whisper" OFF)
option(ENABLE_TTS "Enable TTS" OFF)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_STATIC_LIB "Build static library" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)

if (ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
endif()

if (ADDRESS_SANITIZER)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    # set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(RWKV_MOBILE_SRCS
    src/runtime.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    # src/logger.cpp
    src/soc_detect.cpp
)

if (ENABLE_WEBRWKV_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/web_rwkv/web_rwkv_backend.cpp)

    if (IOS)
        set(Rust_CARGO_TARGET "aarch64-apple-ios")
    endif()

    FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5
    )
    FetchContent_MakeAvailable(Corrosion)

    FetchContent_Declare(
        web_rwkv_ffi
        GIT_REPOSITORY https://github.com/MollySophia/web-rwkv-ffi.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(web_rwkv_ffi)

    corrosion_import_crate(MANIFEST_PATH ${web_rwkv_ffi_SOURCE_DIR}/Cargo.toml
        CRATE_TYPES staticlib)

endif()

if (ENABLE_NCNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/ncnn/ncnn_rwkv_backend.cpp)

    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn
        GIT_TAG master
    )
    # include(disable_unused_ncnn_layers)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    FetchContent_MakeAvailable(ncnn)
endif()

if (ENABLE_LLAMACPP_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/llama_cpp/llama_cpp_backend.cpp)
    if (ENABLE_VISION)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/vision/clip.cpp
            src/multimodal/vision/llava.cpp
        )
    endif()

    if (ENABLE_WHISPER)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/whisper/whisper.cpp
        )
    endif()

    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/mollysophia/llama.cpp
        GIT_TAG rwkv-mobile
    )

    option(LLAMA_STANDALONE "" OFF)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
    FetchContent_MakeAvailable(llama_cpp)
endif()

if (ENABLE_QNN_BACKEND)
    file(GLOB QNN_SRCS backends/qnn/rwkv-qualcomm/Utils/*.cpp backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        backends/qnn/rwkv-qualcomm/PAL/src/linux/*.cpp)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} ${QNN_SRCS})
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/qnn/qnn_backend.cpp)
endif()

if (ENABLE_WHISPER OR ENABLE_TTS)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/multimodal/utils/audio.cpp src/multimodal/tts/cosyvoice.cpp)
    message(CHECK_START "Fetching Eigen3")

    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    # note: To disable eigen tests,
    # you should put this code in a add_subdirectory to avoid to change
    # BUILD_TESTING for your own project too since variables are directory
    # scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    FetchContent_MakeAvailable(Eigen)

    FetchContent_Declare(
        libsoxr
        GIT_REPOSITORY https://github.com/chirlu/soxr.git
        GIT_TAG 0.1.3
    )

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/_deps/libsoxr-src/cmake/Modules")
    option(BUILD_SHARED_LIBS "" OFF)
    if (APPLE AND NOT IOS)
        option(WITH_CR32S "" OFF)
        option(WITH_CR64S "" OFF)
    endif()
    FetchContent_MakeAvailable(libsoxr)

    FetchContent_Declare(
        kaldi_native_fbank
        GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank/
        GIT_TAG master
    )
    option(KALDI_NATIVE_FBANK_BUILD_PYTHON "" OFF)
    option(KALDI_NATIVE_FBANK_BUILD_TESTS "" OFF)
    FetchContent_MakeAvailable(kaldi_native_fbank)

    FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c
        GIT_TAG cpp-6.1.0
        SYSTEM
    )
    option(MSGPACK_USE_BOOST "" OFF)
    FetchContent_MakeAvailable(msgpack)

    # FetchContent_Declare(
    #     WeTextProcessing
    #     GIT_REPOSITORY https://github.com/MollySophia/WeTextProcessing
    #     GIT_TAG master
    # )
    # FetchContent_MakeAvailable(WeTextProcessing)
    # set(HAVE_EXECINFO_H 0)
    # add_subdirectory(${CMAKE_BINARY_DIR}/_deps/wetextprocessing-src/runtime)
endif()

add_library(rwkv_mobile_internal STATIC ${RWKV_MOBILE_SRCS})
target_include_directories(rwkv_mobile_internal PUBLIC src)

if (ENABLE_WHISPER OR ENABLE_TTS)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_TTS)
    target_compile_definitions(rwkv_mobile_internal PUBLIC MSGPACK_NO_BOOST)
    target_link_libraries(rwkv_mobile_internal PUBLIC Eigen3::Eigen msgpack-cxx)
    target_link_libraries(rwkv_mobile_internal PRIVATE $<TARGET_OBJECTS:soxr>)
    if (ENABLE_TTS)
        if (ANDROID)
            target_link_directories(rwkv_mobile_internal PUBLIC src/multimodal/utils/onnxruntime/obj/aarch64-android/)
        elseif (LINUX)
            target_link_directories(rwkv_mobile_internal PUBLIC src/multimodal/utils/onnxruntime/obj/x86_64-linux/)
        endif()
        target_link_libraries(rwkv_mobile_internal PUBLIC onnxruntime $<TARGET_OBJECTS:kaldi-native-fbank-core>
            # wetext_utils wetext_processor
        )
        if (ANDROID)
            target_link_libraries(rwkv_mobile_internal PUBLIC onnxruntime_providers_nnapi)
        endif()
        target_include_directories(rwkv_mobile_internal PUBLIC ${kaldi_native_fbank_SOURCE_DIR})
    endif()
    target_include_directories(rwkv_mobile_internal PUBLIC ${Eigen_SOURCE_DIR} ${libsoxr_SOURCE_DIR}/src 
        src/multimodal/tts src/multimodal/utils 
        src/multimodal/utils/onnxruntime/include
        ${msgpack_SOURCE_DIR}/include)
endif()

if (ENABLE_WEBRWKV_BACKEND)
    if (APPLE)
        set(WEBRWKV_EXTRA_LIBS "-framework QuartzCore -framework Metal -framework CoreGraphics -framework CoreFoundation -lobjc -liconv")
    elseif (WIN32)
        set(WEBRWKV_EXTRA_LIBS ws2_32 opengl32 d3d12 d3dcompiler userenv kernel32 user32 ntdll bcrypt)
    endif()
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WEBRWKV)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include)
    if (IOS)
        target_compile_options(rwkv_mobile_internal PUBLIC "-I${web_rwkv_ffi_SOURCE_DIR}/include")
    endif()
    target_link_libraries(rwkv_mobile_internal PUBLIC web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
endif()

if (ENABLE_NCNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_NCNN)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/ncnn)
    target_link_libraries(rwkv_mobile_internal PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
endif()

if (ENABLE_LLAMACPP_BACKEND)
    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
        endif()
    endif()

    if (GGML_METAL)
        find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
        find_library(METAL_FRAMEWORK    Metal      REQUIRED)
        find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
        add_compile_definitions(GGML_USE_METAL)

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS}
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
        )

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-metal> $<TARGET_OBJECTS:ggml-blas>)
    endif()

    if (ENABLE_VISION)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/vision)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_VISION)
    endif()

    if (ENABLE_WHISPER)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/whisper src/multimodal/utils)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WHISPER)
    endif()

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_LLAMACPP)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/llama_cpp ${llama_cpp_SOURCE_DIR}/include ${llama_cpp_SOURCE_DIR}/src ${llama_cpp_SOURCE_DIR}/ggml/include)
    target_link_libraries(rwkv_mobile_internal PUBLIC $<TARGET_OBJECTS:llama> $<TARGET_OBJECTS:ggml> $<TARGET_OBJECTS:ggml-base> $<TARGET_OBJECTS:ggml-cpu> ${LLAMACPP_EXTRA_LIBS})
endif()

if (ENABLE_QNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_QNN)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/qnn
                               backends/qnn/rwkv-qualcomm
                               backends/qnn/rwkv-qualcomm/Utils
                               backends/qnn/rwkv-qualcomm/PAL/include
                               backends/qnn/rwkv-qualcomm/WrapperUtils
                               backends/qnn/rwkv-qualcomm/QNN/)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (NOT WIN32)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive")
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
endif()

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(rwkv_mobile_internal PUBLIC ${log-lib} ${android-lib})
endif()

if (BUILD_STATIC_LIB)
    add_library(rwkv_mobile_static STATIC src/c_api.cpp)
    target_include_directories(rwkv_mobile_static PUBLIC src)
    target_link_libraries(rwkv_mobile_static PRIVATE rwkv_mobile_internal)

    set (STATIC_LIBS_LIST $<TARGET_FILE:rwkv_mobile_static> $<TARGET_FILE:rwkv_mobile_internal>)

    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile_static PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} ${CMAKE_BINARY_DIR}/Release/cargo/build/${Rust_CARGO_TARGET}/release/libweb_rwkv_ffi.a)
    endif()

    if (ENABLE_WHISPER OR ENABLE_TTS)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:soxr>)
    endif()

    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/librwkv_mobile.a
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/librwkv_mobile.a ${STATIC_LIBS_LIST}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Combining static libraries for rwkv_mobile"
    )
    add_custom_target(rwkv_mobile ALL DEPENDS ${CMAKE_BINARY_DIR}/librwkv_mobile.a)
else()
    add_library(rwkv_mobile SHARED src/c_api.cpp)
    target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rwkv_mobile_internal>")
    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()
endif()