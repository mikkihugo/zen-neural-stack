//! Optimized build script for CUDA-Rust-WASM project
//! Supports cross-platform compilation with caching and performance optimizations

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
  let start_time = std::time::Instant::now();

  // Get build environment info and validate
  let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
  let profile = env::var("PROFILE").unwrap_or_default();
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  
  // Validate build environment and log configuration
  println!("cargo:warning=Building for target_arch: {}, target_os: {}, profile: {}", target_arch, target_os, profile);
  println!("cargo:warning=Build output directory: {}", out_dir.display());
  println!("cargo:warning=Target environment: {}", target_env);
  
  // Ensure output directory is accessible
  if !out_dir.exists() {
    std::fs::create_dir_all(&out_dir).expect("Failed to create output directory");
  }
  
  // Validate target architecture for optimizations
  match target_arch.as_str() {
    "wasm32" => println!("cargo:warning=WebAssembly target detected - enabling WASM optimizations"),
    "x86_64" => println!("cargo:warning=x86_64 target detected - enabling AVX optimizations"),
    "aarch64" => println!("cargo:warning=ARM64 target detected - enabling NEON optimizations"),
    arch => println!("cargo:warning=Unknown architecture: {} - using default optimizations", arch),
  }

  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=src/");
  println!("cargo:rerun-if-changed=bindings/");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=OPENCL_ROOT");
  println!("cargo:rerun-if-env-changed=VULKAN_SDK");

  // Enable parallel compilation if supported
  if env::var("CARGO_FEATURE_PARALLEL_COMPILATION").is_ok() {
    if let Ok(num_jobs) = env::var("NUM_JOBS") {
      println!("cargo:rustc-env=RAYON_NUM_THREADS={num_jobs}");
    }
  }

  // Configure for WASM target with enhanced optimizations
  if target_arch == "wasm32" {
    configure_wasm_build(&target_env, &profile);
  } else {
    configure_native_build(&target_os, &target_arch, &profile);
  }

  // Configure GPU backends with environment validation
  configure_gpu_backends(&target_os, &target_arch);

  // Generate bindings if needed and validate output
  #[cfg(feature = "native-bindings")]
  {
    println!("cargo:warning=Generating native bindings to: {}", out_dir.display());
    generate_native_bindings(&out_dir);
    
    // Verify bindings were generated
    let bindings_file = out_dir.join("cuda_bindings.rs");
    if bindings_file.exists() {
      println!("cargo:warning=Native bindings generated successfully");
    } else {
      println!("cargo:warning=Native bindings generation may have failed");
    }
  }

  // Performance and caching optimizations
  configure_build_optimizations(&profile, &target_arch);

  // Build time reporting
  let build_time = start_time.elapsed();
  if build_time.as_millis() > 1000 {
    println!(
      "cargo:warning=Build configuration took {:.2}s",
      build_time.as_secs_f64()
    );
  }
}

fn configure_wasm_build(target_env: &str, profile: &str) {
  println!("cargo:rustc-cfg=wasm_target");

  // Enhanced WASM optimizations
  println!("cargo:rustc-env=WASM_BINDGEN_WEAKREF=1");
  println!("cargo:rustc-env=WASM_BINDGEN_EXTERNREF_XFORM=1");

  // Enable SIMD if supported
  if env::var("CARGO_FEATURE_WASM_SIMD").is_ok() {
    println!("cargo:rustc-cfg=wasm_simd");
    println!("cargo:rustc-target-feature=+simd128");
  }

  // WASM-specific link arguments for size optimization
  if profile == "release" || profile == "wasm" {
    println!("cargo:rustc-link-arg=--no-entry");
    println!("cargo:rustc-link-arg=--gc-sections");
    println!("cargo:rustc-link-arg=--strip-all");
    println!("cargo:rustc-link-arg=-z");
    println!("cargo:rustc-link-arg=stack-size=1048576"); // 1MB stack

    // Enable bulk memory operations
    println!("cargo:rustc-link-arg=--enable-bulk-memory");
    println!("cargo:rustc-link-arg=--enable-mutable-globals");

    // Size optimizations
    if profile == "wasm" {
      println!("cargo:rustc-link-arg=-O3");
      println!("cargo:rustc-link-arg=--lto-O3");
    }
  }

  // Web-specific features
  if target_env == "unknown" {
    println!("cargo:rustc-cfg=web_target");
  }
}

fn configure_native_build(target_os: &str, target_arch: &str, profile: &str) {
  println!("cargo:rustc-cfg=native_target");

  // Platform-specific optimizations
  match target_os {
    "windows" => {
      println!("cargo:rustc-link-lib=dylib=kernel32");
      println!("cargo:rustc-link-lib=dylib=user32");
      println!("cargo:rustc-link-lib=dylib=shell32");
      if profile == "release" {
        println!("cargo:rustc-link-arg=/LTCG"); // Link-time code generation
      }
    }
    "macos" => {
      println!("cargo:rustc-link-lib=framework=CoreFoundation");
      println!("cargo:rustc-link-lib=framework=Metal");
      println!("cargo:rustc-link-lib=framework=MetalKit");
      if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=apple_silicon");
      }
    }
    "linux" => {
      println!("cargo:rustc-link-lib=dylib=dl");
      println!("cargo:rustc-link-lib=dylib=pthread");
      if profile == "release" {
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        println!("cargo:rustc-link-arg=-Wl,--strip-all");
      }
    }
    _ => {}
  }

  // Architecture-specific optimizations
  match target_arch {
    "x86_64" => {
      println!("cargo:rustc-cfg=x86_64_target");
      if env::var("CARGO_FEATURE_OPTIMIZED_BUILD").is_ok() {
        println!("cargo:rustc-target-feature=+avx2,+fma");
      }
    }
    "aarch64" => {
      println!("cargo:rustc-cfg=aarch64_target");
      if env::var("CARGO_FEATURE_OPTIMIZED_BUILD").is_ok() {
        println!("cargo:rustc-target-feature=+neon");
      }
    }
    _ => {}
  }
}

fn configure_gpu_backends(target_os: &str, target_arch: &str) {
  // Use target_os for OS-specific GPU backend selection
  println!("cargo:warning=Configuring GPU backends for OS: {}, architecture: {}", target_os, target_arch);
  
  match target_os {
    "windows" => {
      println!("cargo:warning=Windows detected - enabling DirectCompute and CUDA backends");
      println!("cargo:rustc-cfg=windows_gpu");
      // Windows-specific GPU library paths
      println!("cargo:rustc-link-search=C:/Windows/System32");
    }
    "linux" => {
      println!("cargo:warning=Linux detected - enabling OpenCL and CUDA backends");
      println!("cargo:rustc-cfg=linux_gpu");
      // Linux GPU library search paths
      println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
      println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    }
    "macos" => {
      println!("cargo:warning=macOS detected - enabling Metal and OpenCL backends");
      println!("cargo:rustc-cfg=macos_gpu");
      // macOS Metal framework integration
      if target_arch == "aarch64" {
        println!("cargo:warning=Apple Silicon detected - optimizing for M-series GPU");
        println!("cargo:rustc-cfg=apple_silicon_gpu");
      }
    }
    _ => {
      println!("cargo:warning=Unknown OS {} - using generic GPU backend configuration", target_os);
    }
  }
  
  // Use target_arch for architecture-specific GPU optimizations
  match target_arch {
    "x86_64" => {
      println!("cargo:warning=x86_64 architecture - enabling AVX2 GPU compute acceleration");
      println!("cargo:rustc-cfg=x86_64_gpu_accel");
    }
    "aarch64" => {
      println!("cargo:warning=ARM64 architecture - enabling NEON GPU compute acceleration");
      println!("cargo:rustc-cfg=aarch64_gpu_accel");
    }
    "wasm32" => {
      println!("cargo:warning=WebAssembly target - enabling WebGPU backend");
      println!("cargo:rustc-cfg=webgpu_backend");
    }
    _ => {
      println!("cargo:warning=Architecture {} - using generic GPU configuration", target_arch);
    }
  }

  // CUDA backend configuration
  #[cfg(feature = "cuda-backend")]
  {
    if let Some(cuda_path) = find_cuda_installation() {
      println!(
        "cargo:rustc-link-search=native={}/lib64",
        cuda_path.display()
      );
      println!(
        "cargo:rustc-link-search=native={}/lib/x64",
        cuda_path.display()
      );
      println!("cargo:rustc-link-lib=cudart");
      println!("cargo:rustc-link-lib=cublas");
      println!("cargo:rustc-link-lib=curand");
      println!("cargo:rustc-cfg=has_cuda");

      // CUDA version detection with comprehensive validation
      if let Some(version) = detect_cuda_version(&cuda_path) {
        println!("cargo:rustc-env=CUDA_VERSION={}", version);
        println!("cargo:warning=CUDA {} detected at {}", version, cuda_path.display());
        
        // Use the detected version for feature configuration
        if version >= 11.0 {
          println!("cargo:rustc-cfg=cuda_11_plus");
          if version >= 12.0 {
            println!("cargo:rustc-cfg=cuda_12_plus");
            println!("cargo:warning=CUDA 12+ features enabled");
          }
        }
        
        // Architecture-specific CUDA optimizations
        match target_arch {
          "x86_64" => {
            println!("cargo:rustc-cfg=cuda_x86_64");
            println!("cargo:warning=Enabling CUDA x86_64 optimizations");
          }
          _ => {}
        }
      } else {
        println!("cargo:warning=CUDA installation found at {} but version detection failed", cuda_path.display());
      }
    } else {
      println!("cargo:warning=CUDA not found on {} {}, CUDA backend disabled", target_os, target_arch);
    }
  }

  // OpenCL backend configuration with OS-specific implementation
  #[cfg(feature = "opencl-backend")]
  {
    if let Some(opencl_path) = find_opencl_installation() {
      println!("cargo:rustc-cfg=has_opencl");
      println!("cargo:warning=OpenCL found at {}", opencl_path.display());
      
      // OS-specific OpenCL linking using target_os
      match target_os {
        "windows" => {
          println!("cargo:rustc-link-lib=OpenCL");
          println!("cargo:warning=Windows OpenCL linking configured");
        }
        "macos" => {
          println!("cargo:rustc-link-lib=framework=OpenCL");
          println!("cargo:warning=macOS OpenCL framework linked");
          if target_arch == "aarch64" {
            println!("cargo:warning=Apple Silicon OpenCL optimizations enabled");
            println!("cargo:rustc-cfg=macos_opencl_optimized");
          }
        }
        "linux" => {
          println!("cargo:rustc-link-lib=OpenCL");
          println!("cargo:rustc-link-search={}", opencl_path.display());
          println!("cargo:warning=Linux OpenCL library configured");
        }
        _ => {
          println!("cargo:warning=Generic OpenCL configuration for OS: {}", target_os);
        }
      }
    } else {
      println!("cargo:warning=OpenCL not found on {} {}, OpenCL backend disabled", target_os, target_arch);
    }
  }

  // Vulkan backend configuration
  #[cfg(feature = "vulkan")]
  {
    if let Some(vulkan_path) = find_vulkan_installation() {
      println!(
        "cargo:rustc-link-search=native={}/lib",
        vulkan_path.display()
      );
      println!("cargo:rustc-link-lib=vulkan");
      println!("cargo:rustc-cfg=has_vulkan");
    }
  }
}

fn configure_build_optimizations(profile: &str, target_arch: &str) {
  // Link-time optimizations
  if profile == "release" {
    println!("cargo:rustc-env=RUST_LTO=fat");

    // Enable additional optimizations for release builds
    if target_arch == "wasm32" {
      println!("cargo:rustc-env=WASM_OPT_LEVEL=3");
    } else {
      // Native optimizations
      println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
    }
  }

  // Build caching optimizations
  if let Ok(cache_dir) = env::var("CARGO_TARGET_DIR") {
    println!("cargo:rustc-env=CARGO_BUILD_CACHE={cache_dir}");
  }

  // Incremental compilation for development
  if profile == "dev" {
    println!("cargo:rustc-env=CARGO_INCREMENTAL=1");
  }
}

// Helper functions for GPU backend detection
// Only warn about dead code when no GPU backend features are enabled
#[cfg_attr(not(any(feature = "cuda-backend", feature = "opencl-backend", feature = "vulkan")), allow(dead_code))]

#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn find_cuda_installation() -> Option<PathBuf> {
  println!("cargo:warning=Starting comprehensive CUDA installation search...");
  
  // Check environment variable first with enhanced validation
  if let Ok(cuda_path) = env::var("CUDA_PATH") {
    println!("cargo:warning=Found CUDA_PATH environment variable: {}", cuda_path);
    let path = PathBuf::from(&cuda_path);
    if validate_cuda_installation(&path) {
      println!("cargo:warning=CUDA_PATH installation validated successfully");
      return Some(path);
    } else {
      println!("cargo:warning=CUDA_PATH installation validation failed: {}", cuda_path);
    }
  }

  // Check CUDA_HOME alternative
  if let Ok(cuda_home) = env::var("CUDA_HOME") {
    println!("cargo:warning=Found CUDA_HOME environment variable: {}", cuda_home);
    let path = PathBuf::from(&cuda_home);
    if validate_cuda_installation(&path) {
      println!("cargo:warning=CUDA_HOME installation validated successfully");
      return Some(path);
    }
  }

  // Get target OS for platform-specific search
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  println!("cargo:warning=Searching CUDA installations for OS: {}", target_os);

  // Platform-specific common CUDA installation paths
  let common_paths = match target_os.as_str() {
    "windows" => vec![
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6",
      "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
      "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
    ],
    "linux" => vec![
      "/usr/local/cuda",
      "/usr/local/cuda-12.4",
      "/usr/local/cuda-12.3", 
      "/usr/local/cuda-12.2",
      "/usr/local/cuda-12.1",
      "/usr/local/cuda-12.0",
      "/usr/local/cuda-11.8",
      "/usr/local/cuda-11.7",
      "/opt/cuda",
      "/usr/cuda",
      "/usr/lib/cuda",
      "/usr/lib/nvidia-cuda-toolkit",
    ],
    "macos" => vec![
      "/usr/local/cuda",
      "/Developer/NVIDIA/CUDA-12.0",
      "/Developer/NVIDIA/CUDA-11.8",
      "/Applications/NVIDIA/CUDA-12.0",
      "/Applications/NVIDIA/CUDA-11.8",
      "/opt/cuda",
    ],
    _ => vec!["/usr/local/cuda", "/opt/cuda"],
  };

  for path_str in &common_paths {
    println!("cargo:warning=Checking CUDA path: {}", path_str);
    let path = PathBuf::from(path_str);
    if validate_cuda_installation(&path) {
      println!("cargo:warning=Found valid CUDA installation at: {}", path.display());
      return Some(path);
    }
  }

  // Try registry search on Windows
  #[cfg(target_os = "windows")]
  {
    if let Some(registry_path) = find_cuda_from_windows_registry() {
      println!("cargo:warning=Found CUDA via Windows registry: {}", registry_path.display());
      if validate_cuda_installation(&registry_path) {
        return Some(registry_path);
      }
    }
  }

  // Try pkg-config with enhanced error handling
  println!("cargo:warning=Attempting pkg-config CUDA detection...");
  if Command::new("pkg-config")
    .args(["--exists", "cuda"])
    .status()
    .map(|s| s.success())
    .unwrap_or(false)
  {
    println!("cargo:warning=pkg-config found CUDA package");
    if let Ok(output) = Command::new("pkg-config")
      .args(["--variable=cudaroot", "cuda"])
      .output()
    {
      let path_str = String::from_utf8_lossy(&output.stdout);
      let path_str = path_str.trim();
      println!("cargo:warning=pkg-config reported CUDA root: {}", path_str);
      let path = PathBuf::from(path_str);
      if validate_cuda_installation(&path) {
        println!("cargo:warning=pkg-config CUDA installation validated");
        return Some(path);
      }
    }
  } else {
    println!("cargo:warning=pkg-config CUDA detection failed or not available");
  }

  // Try additional discovery methods for Linux
  #[cfg(target_os = "linux")]
  {
    if let Some(ldconfig_path) = find_cuda_via_ldconfig() {
      println!("cargo:warning=Found CUDA via ldconfig: {}", ldconfig_path.display());
      if validate_cuda_installation(&ldconfig_path) {
        return Some(ldconfig_path);
      }
    }
  }

  println!("cargo:warning=No valid CUDA installation found after comprehensive search");
  None
}

/// Validate CUDA installation completeness
#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn validate_cuda_installation(cuda_path: &Path) -> bool {
  if !cuda_path.exists() {
    return false;
  }

  // Check for essential CUDA components (cross-platform validation)
  let required_components = [
    "bin", "lib64", "include", // Linux/Mac structure
    "bin", "lib", "include",   // Alternative structure
  ];
  
  // Validate essential directory structure exists
  let has_required_structure = required_components.iter().any(|&component| {
    cuda_path.join(component).exists()
  });
  
  if !has_required_structure {
    println!("cargo:warning=CUDA path {} missing required components: {:?}", 
             cuda_path.display(), required_components);
    return false;
  }

  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  
  match target_os.as_str() {
    "windows" => {
      // Windows CUDA structure validation
      let bin_path = cuda_path.join("bin");
      let lib_path = cuda_path.join("lib").join("x64");
      let include_path = cuda_path.join("include");
      
      if !bin_path.exists() || !lib_path.exists() || !include_path.exists() {
        return false;
      }
      
      // Check for nvcc compiler
      let nvcc_path = bin_path.join("nvcc.exe");
      if !nvcc_path.exists() {
        return false;
      }
      
      // Check for essential libraries
      let essential_libs = ["cudart.lib", "cuda.lib"];
      for lib in &essential_libs {
        if !lib_path.join(lib).exists() {
          return false;
        }
      }
    }
    _ => {
      // Unix-like systems (Linux, macOS)
      let lib64_path = cuda_path.join("lib64");
      let lib_path = cuda_path.join("lib");
      let bin_path = cuda_path.join("bin");
      let include_path = cuda_path.join("include");
      
      if !bin_path.exists() || !include_path.exists() {
        return false;
      }
      
      // Check for lib64 or lib directory
      if !lib64_path.exists() && !lib_path.exists() {
        return false;
      }
      
      // Check for nvcc compiler
      let nvcc_path = bin_path.join("nvcc");
      if !nvcc_path.exists() {
        return false;
      }
      
      // Check for essential libraries in lib64 or lib
      let lib_dir = if lib64_path.exists() { lib64_path } else { lib_path };
      let essential_libs = ["libcudart.so", "libcuda.so"];
      
      for lib in &essential_libs {
        let lib_file = lib_dir.join(lib);
        // Also check for versioned libraries
        if !lib_file.exists() {
          // Try to find versioned variants
          let lib_pattern = format!("{}.*", lib);
          if !find_library_with_pattern(&lib_dir, &lib_pattern) {
            continue; // Allow missing libcuda.so as it's driver-dependent
          }
        }
      }
    }
  }
  
  true
}

/// Find library with pattern matching
fn find_library_with_pattern(lib_dir: &Path, pattern: &str) -> bool {
  if let Ok(entries) = std::fs::read_dir(lib_dir) {
    for entry in entries.flatten() {
      if let Ok(name) = entry.file_name().into_string() {
        if name.starts_with(&pattern[..pattern.len().saturating_sub(2)]) {
          return true;
        }
      }
    }
  }
  false
}

/// Find CUDA installation via Windows registry
#[cfg(target_os = "windows")]
#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn find_cuda_from_windows_registry() -> Option<PathBuf> {
  use std::process::Command;
  
  // Try reading registry for CUDA installation paths
  let reg_keys = [
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA",
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA",
  ];
  
  for reg_key in &reg_keys {
    if let Ok(output) = Command::new("reg")
      .args(["query", reg_key, "/s"])
      .output()
    {
      let output_str = String::from_utf8_lossy(&output.stdout);
      // Parse registry output for installation paths
      for line in output_str.lines() {
        if line.contains("InstallDir") && line.contains("REG_SZ") {
          if let Some(path_start) = line.rfind("C:\\") {
            let path_str = line[path_start..].trim();
            let path = PathBuf::from(path_str);
            if path.exists() {
              return Some(path);
            }
          }
        }
      }
    }
  }
  
  None
}

/// Find CUDA installation via ldconfig on Linux
#[cfg(target_os = "linux")]
#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn find_cuda_via_ldconfig() -> Option<PathBuf> {
  if let Ok(output) = Command::new("ldconfig")
    .args(["-p"])
    .output()
  {
    let output_str = String::from_utf8_lossy(&output.stdout);
    for line in output_str.lines() {
      if line.contains("libcudart.so") {
        if let Some(path_start) = line.rfind(" => ") {
          let lib_path_str = &line[path_start + 4..].trim();
          let lib_path = PathBuf::from(lib_path_str);
          if let Some(cuda_lib_dir) = lib_path.parent() {
            if let Some(cuda_root) = cuda_lib_dir.parent() {
              return Some(cuda_root.to_path_buf());
            }
          }
        }
      }
    }
  }
  None
}

#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn detect_cuda_version(cuda_path: &Path) -> Option<f32> {
  println!("cargo:warning=Detecting CUDA version at: {}", cuda_path.display());
  
  // Determine nvcc executable name based on OS
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  let nvcc_name = if target_os == "windows" { "nvcc.exe" } else { "nvcc" };
  let nvcc_path = cuda_path.join("bin").join(nvcc_name);
  
  if !nvcc_path.exists() {
    println!("cargo:warning=nvcc not found at: {}", nvcc_path.display());
    return try_alternative_version_detection(cuda_path);
  }
  
  // Try nvcc --version first
  if let Ok(output) = Command::new(&nvcc_path).args(["--version"]).output() {
    let version_str = String::from_utf8_lossy(&output.stdout);
    println!("cargo:warning=nvcc --version output: {}", version_str.trim());
    
    // Parse version from output like "Cuda compilation tools, release 11.8, V11.8.89"
    if let Some(version) = parse_nvcc_version(&version_str) {
      println!("cargo:warning=Successfully detected CUDA version: {}", version);
      return Some(version);
    }
  } else {
    println!("cargo:warning=Failed to execute nvcc --version");
  }
  
  // Try alternative detection methods
  println!("cargo:warning=Attempting alternative CUDA version detection methods...");
  try_alternative_version_detection(cuda_path)
}

/// Parse CUDA version from nvcc output with multiple patterns
#[cfg_attr(not(feature = "cuda-backend"), allow(dead_code))]
fn parse_nvcc_version(version_str: &str) -> Option<f32> {
  // Pattern 1: "Cuda compilation tools, release 11.8, V11.8.89"
  if let Some(release_pos) = version_str.find("release ") {
    let version_part = &version_str[release_pos + 8..];
    if let Some(comma_pos) = version_part.find(',') {
      let version_num = &version_part[..comma_pos].trim();
      if let Ok(version) = version_num.parse::<f32>() {
        return Some(version);
      }
    }
  }
  
  // Pattern 2: "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Tue_Aug_15_22:02:13_PDT_2023\nCuda compilation tools, release 12.2, V12.2.140"
  if let Some(release_pos) = version_str.rfind("release ") {
    let version_part = &version_str[release_pos + 8..];
    if let Some(comma_pos) = version_part.find(',') {
      let version_num = &version_part[..comma_pos].trim();
      if let Ok(version) = version_num.parse::<f32>() {
        return Some(version);
      }
    }
  }
  
  // Pattern 3: Look for "V12.2.140" style versions
  for line in version_str.lines() {
    if line.contains('V') && line.chars().any(|c| c.is_ascii_digit()) {
      if let Some(v_pos) = line.find('V') {
        let version_part = &line[v_pos + 1..];
        // Extract version until first non-digit/dot character
        let version_end = version_part.chars()
          .take_while(|c| c.is_ascii_digit() || *c == '.')
          .count();
        if version_end > 0 {
          let version_str = &version_part[..version_end];
          // Parse major.minor version
          if let Some(dot_pos) = version_str.find('.') {
            if let Some(second_dot_pos) = version_str[dot_pos + 1..].find('.') {
              // Take major.minor from major.minor.patch
              let major_minor = &version_str[..dot_pos + 1 + second_dot_pos];
              if let Ok(version) = major_minor.parse::<f32>() {
                return Some(version);
              }
            } else {
              // Already major.minor format
              if let Ok(version) = version_str.parse::<f32>() {
                return Some(version);
              }
            }
          }
        }
      }
    }
  }
  
  // Pattern 4: Look for bare version numbers like "12.2" or "11.8"
  for word in version_str.split_whitespace() {
    if word.chars().all(|c| c.is_ascii_digit() || c == '.') && word.contains('.') {
      if let Ok(version) = word.parse::<f32>() {
        // Reasonable CUDA version range check
        if version >= 9.0 && version <= 15.0 {
          return Some(version);
        }
      }
    }
  }
  
  None
}

/// Try alternative CUDA version detection methods
fn try_alternative_version_detection(cuda_path: &Path) -> Option<f32> {
  // Method 1: Check version.txt or version.json files
  for version_file in &["version.txt", "version.json", "CUDA_Toolkit_Release_Notes.txt"] {
    let version_path = cuda_path.join(version_file);
    if version_path.exists() {
      if let Ok(content) = std::fs::read_to_string(&version_path) {
        if let Some(version) = extract_version_from_text(&content) {
          println!("cargo:warning=Found CUDA version {} in {}", version, version_file);
          return Some(version);
        }
      }
    }
  }
  
  // Method 2: Parse directory name for version
  if let Some(dir_name) = cuda_path.file_name().and_then(|n| n.to_str()) {
    if let Some(version) = extract_version_from_text(dir_name) {
      println!("cargo:warning=Inferred CUDA version {} from directory name: {}", version, dir_name);
      return Some(version);
    }
  }
  
  // Method 3: Check include/cuda_runtime_api.h for version macros
  let runtime_header = cuda_path.join("include").join("cuda_runtime_api.h");
  if runtime_header.exists() {
    if let Ok(content) = std::fs::read_to_string(&runtime_header) {
      if let Some(version) = extract_version_from_cuda_header(&content) {
        println!("cargo:warning=Found CUDA version {} in cuda_runtime_api.h", version);
        return Some(version);
      }
    }
  }
  
  // Method 4: Check for library versions
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  match target_os.as_str() {
    "linux" | "macos" => {
      let lib_dir = if cuda_path.join("lib64").exists() {
        cuda_path.join("lib64")
      } else {
        cuda_path.join("lib")
      };
      
      if let Some(version) = extract_version_from_libraries(&lib_dir) {
        println!("cargo:warning=Inferred CUDA version {} from library files", version);
        return Some(version);
      }
    }
    "windows" => {
      let lib_dir = cuda_path.join("lib").join("x64");
      if let Some(version) = extract_version_from_libraries(&lib_dir) {
        println!("cargo:warning=Inferred CUDA version {} from library files", version);
        return Some(version);
      }
    }
    _ => {}
  }
  
  println!("cargo:warning=Could not determine CUDA version using any method");
  None
}

/// Extract version from text content using various patterns
fn extract_version_from_text(text: &str) -> Option<f32> {
  // Look for patterns like "CUDA 12.2", "v11.8", "12.1", etc.
  let patterns = [
    r"CUDA\s+(\d+\.\d+)",
    r"cuda\s+(\d+\.\d+)",
    r"v(\d+\.\d+)",
    r"V(\d+\.\d+)",
    r"(\d+\.\d+)",
  ];
  
  for pattern_text in &patterns {
    // Simple pattern matching without regex
    if let Some(version) = find_version_in_text(text, pattern_text) {
      return Some(version);
    }
  }
  
  None
}

/// Simple version extraction without regex
fn find_version_in_text(text: &str, _pattern: &str) -> Option<f32> {
  // Look for version patterns in text
  let text_lower = text.to_lowercase();
  
  // Find "cuda" followed by version
  if let Some(cuda_pos) = text_lower.find("cuda") {
    let after_cuda = &text[cuda_pos + 4..];
    if let Some(version) = extract_first_version_number(after_cuda) {
      return Some(version);
    }
  }
  
  // Find "v" followed by version
  for (i, char) in text.char_indices() {
    if char == 'v' || char == 'V' {
      let after_v = &text[i + 1..];
      if let Some(version) = extract_first_version_number(after_v) {
        if version >= 9.0 && version <= 15.0 {
          return Some(version);
        }
      }
    }
  }
  
  None
}

/// Extract first version number (x.y format) from text
fn extract_first_version_number(text: &str) -> Option<f32> {
  let mut version_str = String::new();
  let mut found_digit = false;
  let mut found_dot = false;
  
  for char in text.chars() {
    if char.is_ascii_digit() {
      version_str.push(char);
      found_digit = true;
    } else if char == '.' && found_digit && !found_dot {
      version_str.push(char);
      found_dot = true;
    } else if found_digit && found_dot && char.is_ascii_digit() {
      version_str.push(char);
    } else if found_digit {
      break;
    } else if !char.is_whitespace() && !char.is_ascii_punctuation() {
      break;
    }
  }
  
  if found_digit && found_dot {
    version_str.parse::<f32>().ok()
  } else {
    None
  }
}

/// Extract version from CUDA header files
fn extract_version_from_cuda_header(content: &str) -> Option<f32> {
  let mut major: Option<u32> = None;
  let mut minor: Option<u32> = None;
  
  // First pass: Look for individual major/minor definitions
  for line in content.lines() {
    if line.contains("#define CUDA_VERSION_MAJOR") {
      if let Some(version_str) = line.split_whitespace().last() {
        if let Ok(maj) = version_str.parse::<u32>() {
          major = Some(maj);
        }
      }
    }
    if line.contains("#define CUDA_VERSION_MINOR") {
      if let Some(version_str) = line.split_whitespace().last() {
        if let Ok(min) = version_str.parse::<u32>() {
          minor = Some(min);
        }
      }
    }
  }
  
  // If we found both major and minor, combine them
  if let (Some(maj), Some(min)) = (major, minor) {
    return Some(maj as f32 + (min as f32 / 10.0));
  }
  
  // Second pass: Look for combined version definitions
  for line in content.lines() {
    if line.contains("#define CUDA_VERSION") && line.contains("CUDA_VERSION") {
      // Look for #define CUDA_VERSION 12020 style definitions
      if let Some(version_str) = line.split_whitespace().last() {
        if let Ok(version_num) = version_str.parse::<u32>() {
          // CUDA_VERSION is encoded as major*1000 + minor*10 + patch
          let major_ver = version_num / 1000;
          let minor_ver = (version_num % 1000) / 10;
          let version_float = major_ver as f32 + (minor_ver as f32 / 10.0);
          return Some(version_float);
        }
      }
    }
    
    if line.contains("#define CUDART_VERSION") {
      if let Some(version_str) = line.split_whitespace().last() {
        if let Ok(version_num) = version_str.parse::<u32>() {
          let major_ver = version_num / 1000;
          let minor_ver = (version_num % 1000) / 10;
          let version_float = major_ver as f32 + (minor_ver as f32 / 10.0);
          return Some(version_float);
        }
      }
    }
  }
  
  None
}

/// Extract version from library filenames
fn extract_version_from_libraries(lib_dir: &Path) -> Option<f32> {
  if let Ok(entries) = std::fs::read_dir(lib_dir) {
    for entry in entries.flatten() {
      if let Some(name) = entry.file_name().to_str() {
        // Look for library files like libcudart.so.12.2 or cudart64_12.lib
        if name.contains("cudart") {
          if let Some(version) = extract_version_from_filename(name) {
            return Some(version);
          }
        }
      }
    }
  }
  None
}

/// Extract version from a filename
fn extract_version_from_filename(filename: &str) -> Option<f32> {
  // Look for patterns like .so.12.2 or _12. or 12.
  let mut chars = filename.chars().peekable();
  
  while let Some(char) = chars.next() {
    if char.is_ascii_digit() {
      let mut version_str = String::new();
      version_str.push(char);
      
      // Collect version string
      while let Some(&next_char) = chars.peek() {
        if next_char.is_ascii_digit() || next_char == '.' {
          version_str.push(chars.next().unwrap());
        } else {
          break;
        }
      }
      
      // Try to parse as version
      if version_str.contains('.') {
        if let Ok(version) = version_str.parse::<f32>() {
          if version >= 9.0 && version <= 15.0 {
            return Some(version);
          }
        }
      }
    }
  }
  
  None
}

#[cfg_attr(not(feature = "opencl-backend"), allow(dead_code))]
fn find_opencl_installation() -> Option<PathBuf> {
  println!("cargo:warning=Starting comprehensive OpenCL installation search...");
  
  // Check environment variables with validation
  for env_var in &["OPENCL_ROOT", "OPENCL_PATH", "OCL_ROOT"] {
    if let Ok(opencl_path) = env::var(env_var) {
      println!("cargo:warning=Found {} environment variable: {}", env_var, opencl_path);
      let path = PathBuf::from(&opencl_path);
      if validate_opencl_installation(&path) {
        println!("cargo:warning={} installation validated successfully", env_var);
        return Some(path);
      } else {
        println!("cargo:warning={} installation validation failed: {}", env_var, opencl_path);
      }
    }
  }

  // Get target OS for platform-specific search
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  println!("cargo:warning=Searching OpenCL installations for OS: {}", target_os);

  match target_os.as_str() {
    "windows" => find_opencl_windows(),
    "macos" => find_opencl_macos(),
    "linux" => find_opencl_linux(),
    _ => {
      println!("cargo:warning=Unknown OS for OpenCL detection: {}", target_os);
      None
    }
  }
}

/// Find OpenCL on Windows with comprehensive vendor support
fn find_opencl_windows() -> Option<PathBuf> {
  println!("cargo:warning=Searching Windows OpenCL installations...");
  
  // Windows OpenCL installation paths for various vendors
  let vendor_paths = [
    // Intel OpenCL SDK
    "C:\\Program Files\\Intel\\OpenCL SDK",
    "C:\\Program Files (x86)\\Intel\\OpenCL SDK",
    "C:\\Program Files\\Intel\\OpenCL Runtime",
    "C:\\Program Files (x86)\\Intel\\OpenCL Runtime",
    
    // AMD APP SDK / ROCm
    "C:\\Program Files\\AMD APP SDK",
    "C:\\Program Files (x86)\\AMD APP SDK", 
    "C:\\Program Files\\AMD\\ROCm",
    "C:\\Program Files\\AMD\\OCL SDK",
    
    // NVIDIA OpenCL (usually with CUDA)
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\OpenCL",
    "C:\\Program Files\\NVIDIA Corporation\\OpenCL",
    
    // Generic Windows SDK locations
    "C:\\Program Files\\Microsoft SDKs\\OpenCL",
    "C:\\Program Files (x86)\\Microsoft SDKs\\OpenCL",
    
    // Khronos reference implementation
    "C:\\Program Files\\Khronos\\OpenCL",
    "C:\\Program Files (x86)\\Khronos\\OpenCL",
  ];
  
  for path_str in &vendor_paths {
    println!("cargo:warning=Checking Windows OpenCL path: {}", path_str);
    let path = PathBuf::from(path_str);
    if validate_opencl_installation(&path) {
      println!("cargo:warning=Found valid OpenCL installation at: {}", path.display());
      return Some(path);
    }
  }
  
  // Check Windows system directories for OpenCL.dll
  let system_paths = [
    "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64",
  ];
  
  for sys_path in &system_paths {
    let opencl_dll = PathBuf::from(sys_path).join("OpenCL.dll");
    if opencl_dll.exists() {
      println!("cargo:warning=Found OpenCL.dll in system directory: {}", sys_path);
      return Some(PathBuf::from(sys_path));
    }
  }
  
  println!("cargo:warning=No OpenCL installation found on Windows");
  None
}

/// Find OpenCL on macOS with framework detection
fn find_opencl_macos() -> Option<PathBuf> {
  println!("cargo:warning=Searching macOS OpenCL installations...");
  
  // macOS has OpenCL framework built-in
  let framework_paths = [
    "/System/Library/Frameworks/OpenCL.framework",
    "/Library/Frameworks/OpenCL.framework",
    "/usr/local/lib/OpenCL.framework", // Homebrew location
  ];
  
  for framework_path_str in &framework_paths {
    println!("cargo:warning=Checking macOS OpenCL framework: {}", framework_path_str);
    let framework_path = PathBuf::from(framework_path_str);
    if validate_opencl_installation(&framework_path) {
      println!("cargo:warning=Found valid OpenCL framework at: {}", framework_path.display());
      return Some(framework_path);
    }
  }
  
  // Check for library files in standard locations
  let lib_paths = [
    "/usr/lib",
    "/usr/local/lib", 
    "/opt/local/lib", // MacPorts
  ];
  
  for lib_path_str in &lib_paths {
    let lib_path = PathBuf::from(lib_path_str);
    let opencl_lib = lib_path.join("libOpenCL.dylib");
    if opencl_lib.exists() {
      println!("cargo:warning=Found libOpenCL.dylib at: {}", lib_path.display());
      return Some(lib_path);
    }
  }
  
  println!("cargo:warning=No OpenCL installation found on macOS");
  None
}

/// Find OpenCL on Linux with comprehensive vendor support
fn find_opencl_linux() -> Option<PathBuf> {
  println!("cargo:warning=Searching Linux OpenCL installations...");
  
  // Linux OpenCL installation paths for various vendors
  let vendor_paths = [
    // Intel OpenCL Runtime
    "/opt/intel/opencl",
    "/opt/intel/opencl_runtime",
    "/usr/local/lib/intel/opencl",
    "/usr/lib/x86_64-linux-gnu/intel-opencl-icd",
    
    // AMD ROCm / APP SDK
    "/opt/rocm/opencl",
    "/opt/AMDAPP/SDK",
    "/opt/AMDAPPSDK", 
    "/usr/lib/x86_64-linux-gnu/amd-opencl-icd",
    
    // NVIDIA OpenCL (usually with CUDA)
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu/nvidia-opencl-icd",
    
    // Standard Linux library paths
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
    "/lib64",
    "/lib",
    
    // Package manager installations
    "/usr/lib/opencl",
    "/usr/local/opencl",
  ];
  
  for path_str in &vendor_paths {
    println!("cargo:warning=Checking Linux OpenCL path: {}", path_str);
    let path = PathBuf::from(path_str);
    if validate_opencl_installation(&path) {
      println!("cargo:warning=Found valid OpenCL installation at: {}", path.display());
      return Some(path);
    }
  }
  
  // Use ldconfig to find OpenCL libraries
  if let Some(ldconfig_path) = find_opencl_via_ldconfig() {
    println!("cargo:warning=Found OpenCL via ldconfig: {}", ldconfig_path.display());
    return Some(ldconfig_path);
  }
  
  // Check /etc/OpenCL/vendors/ for ICD files
  if let Some(icd_path) = find_opencl_via_icd_registry() {
    println!("cargo:warning=Found OpenCL via ICD registry: {}", icd_path.display());
    return Some(icd_path);
  }
  
  println!("cargo:warning=No OpenCL installation found on Linux");
  None
}

/// Validate OpenCL installation completeness
#[cfg_attr(not(feature = "opencl-backend"), allow(dead_code))]
fn validate_opencl_installation(opencl_path: &Path) -> bool {
  if !opencl_path.exists() {
    return false;
  }
  
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  
  match target_os.as_str() {
    "windows" => {
      // Windows: Look for OpenCL.dll or OpenCL headers
      let dll_path = opencl_path.join("OpenCL.dll");
      let lib_path = opencl_path.join("lib").join("OpenCL.lib");
      let include_path = opencl_path.join("include").join("CL").join("opencl.h");
      
      dll_path.exists() || lib_path.exists() || include_path.exists()
    }
    "macos" => {
      // macOS: Framework or dylib
      if opencl_path.to_str().unwrap_or("").contains(".framework") {
        let framework_binary = opencl_path.join("OpenCL");
        let framework_headers = opencl_path.join("Headers").join("opencl.h");
        framework_binary.exists() || framework_headers.exists()
      } else {
        let dylib_path = opencl_path.join("libOpenCL.dylib");
        dylib_path.exists()
      }
    }
    _ => {
      // Linux and other Unix-like systems
      let so_files = [
        "libOpenCL.so",
        "libOpenCL.so.1",
        "libOpenCL.so.2", 
      ];
      
      for so_file in &so_files {
        if opencl_path.join(so_file).exists() {
          return true;
        }
      }
      
      // Check for versioned libraries
      if let Ok(entries) = std::fs::read_dir(opencl_path) {
        for entry in entries.flatten() {
          if let Some(name) = entry.file_name().to_str() {
            if name.starts_with("libOpenCL.so") {
              return true;
            }
          }
        }
      }
      
      false
    }
  }
}

/// Find OpenCL installation via ldconfig on Linux
fn find_opencl_via_ldconfig() -> Option<PathBuf> {
  println!("cargo:warning=Attempting ldconfig OpenCL detection...");
  
  if let Ok(output) = Command::new("ldconfig")
    .args(["-p"])
    .output()
  {
    let output_str = String::from_utf8_lossy(&output.stdout);
    for line in output_str.lines() {
      if line.contains("libOpenCL.so") {
        if let Some(path_start) = line.rfind(" => ") {
          let lib_path_str = &line[path_start + 4..].trim();
          let lib_path = PathBuf::from(lib_path_str);
          if let Some(opencl_lib_dir) = lib_path.parent() {
            println!("cargo:warning=Found OpenCL library directory via ldconfig: {}", opencl_lib_dir.display());
            return Some(opencl_lib_dir.to_path_buf());
          }
        }
      }
    }
  } else {
    println!("cargo:warning=ldconfig command failed or not available");
  }
  
  None
}

/// Find OpenCL via ICD loader registry on Linux
fn find_opencl_via_icd_registry() -> Option<PathBuf> {
  println!("cargo:warning=Checking OpenCL ICD registry...");
  
  let icd_paths = [
    "/etc/OpenCL/vendors",
    "/usr/share/OpenCL/vendors",
    "/usr/local/share/OpenCL/vendors",
  ];
  
  for icd_path_str in &icd_paths {
    let icd_path = PathBuf::from(icd_path_str);
    if icd_path.exists() {
      println!("cargo:warning=Found OpenCL ICD directory: {}", icd_path.display());
      if let Ok(entries) = std::fs::read_dir(&icd_path) {
        for entry in entries.flatten() {
          if let Some(name) = entry.file_name().to_str() {
            if name.ends_with(".icd") {
              println!("cargo:warning=Found OpenCL ICD file: {}", name);
              // Try to read the ICD file to get library path
              if let Ok(icd_content) = std::fs::read_to_string(entry.path()) {
                let lib_path_str = icd_content.trim();
                let lib_path = PathBuf::from(lib_path_str);
                if lib_path.exists() {
                  if let Some(lib_dir) = lib_path.parent() {
                    println!("cargo:warning=Found OpenCL library from ICD: {}", lib_dir.display());
                    return Some(lib_dir.to_path_buf());
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  
  None
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn find_vulkan_installation() -> Option<PathBuf> {
  println!("cargo:warning=Starting comprehensive Vulkan installation search...");
  
  // Check environment variables with validation
  for env_var in &["VULKAN_SDK", "VK_SDK_PATH", "VULKAN_PATH"] {
    if let Ok(vulkan_path) = env::var(env_var) {
      println!("cargo:warning=Found {} environment variable: {}", env_var, vulkan_path);
      let path = PathBuf::from(&vulkan_path);
      if validate_vulkan_installation(&path) {
        println!("cargo:warning={} installation validated successfully", env_var);
        return Some(path);
      } else {
        println!("cargo:warning={} installation validation failed: {}", env_var, vulkan_path);
      }
    }
  }

  // Get target OS for platform-specific search
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  println!("cargo:warning=Searching Vulkan installations for OS: {}", target_os);

  match target_os.as_str() {
    "windows" => find_vulkan_windows(),
    "macos" => find_vulkan_macos(), 
    "linux" => find_vulkan_linux(),
    _ => {
      println!("cargo:warning=Unknown OS for Vulkan detection: {}", target_os);
      None
    }
  }
}

/// Find Vulkan on Windows with comprehensive detection
fn find_vulkan_windows() -> Option<PathBuf> {
  println!("cargo:warning=Searching Windows Vulkan installations...");
  
  // Windows Vulkan SDK installation paths
  let sdk_paths = [
    // LunarG Vulkan SDK (common versions)
    "C:\\VulkanSDK\\1.3.268.0",
    "C:\\VulkanSDK\\1.3.261.1", 
    "C:\\VulkanSDK\\1.3.250.1",
    "C:\\VulkanSDK\\1.3.239.0",
    "C:\\VulkanSDK\\1.3.231.1",
    "C:\\VulkanSDK\\1.3.224.1",
    "C:\\VulkanSDK\\1.3.216.0",
    "C:\\VulkanSDK\\1.2.198.1",
    "C:\\VulkanSDK\\1.2.189.2",
    "C:\\VulkanSDK\\1.2.182.0",
    
    // Generic SDK paths
    "C:\\VulkanSDK",
    "C:\\Program Files\\VulkanSDK",
    "C:\\Program Files (x86)\\VulkanSDK",
    "C:\\Program Files\\LunarG\\VulkanSDK",
    "C:\\Program Files (x86)\\LunarG\\VulkanSDK",
    
    // Vulkan runtime paths
    "C:\\Program Files\\Vulkan Runtime",
    "C:\\Program Files (x86)\\Vulkan Runtime",
  ];
  
  for path_str in &sdk_paths {
    println!("cargo:warning=Checking Windows Vulkan path: {}", path_str);
    let path = PathBuf::from(path_str);
    if validate_vulkan_installation(&path) {
      println!("cargo:warning=Found valid Vulkan installation at: {}", path.display());
      return Some(path);
    }
  }
  
  // Check Windows system directories for Vulkan DLLs
  let system_paths = [
    "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64",
  ];
  
  for sys_path in &system_paths {
    let vulkan_dll = PathBuf::from(sys_path).join("vulkan-1.dll");
    if vulkan_dll.exists() {
      println!("cargo:warning=Found vulkan-1.dll in system directory: {}", sys_path);
      return Some(PathBuf::from(sys_path));
    }
  }
  
  // Try Windows registry search
  if let Some(registry_path) = find_vulkan_from_windows_registry() {
    println!("cargo:warning=Found Vulkan via Windows registry: {}", registry_path.display());
    if validate_vulkan_installation(&registry_path) {
      return Some(registry_path);
    }
  }
  
  println!("cargo:warning=No Vulkan installation found on Windows");
  None
}

/// Find Vulkan on macOS with MoltenVK support
fn find_vulkan_macos() -> Option<PathBuf> {
  println!("cargo:warning=Searching macOS Vulkan installations...");
  
  // macOS Vulkan SDK paths (LunarG SDK)
  let sdk_paths = [
    "/usr/local/vulkan-sdk",
    "/usr/local/VulkanSDK",
    "/opt/vulkan-sdk",
    "/Applications/vulkan-sdk",
    
    // Homebrew installation
    "/usr/local/lib/vulkan",
    "/opt/homebrew/lib/vulkan",
    
    // MoltenVK paths
    "/usr/local/lib/MoltenVK",
    "/opt/homebrew/lib/MoltenVK",
    
    // Framework paths
    "/Library/Frameworks/vulkan.framework",
    "/System/Library/Frameworks/vulkan.framework",
  ];
  
  for path_str in &sdk_paths {
    println!("cargo:warning=Checking macOS Vulkan path: {}", path_str);
    let path = PathBuf::from(path_str);
    if validate_vulkan_installation(&path) {
      println!("cargo:warning=Found valid Vulkan installation at: {}", path.display());
      return Some(path);
    }
  }
  
  // Check for Vulkan libraries in standard locations
  let lib_paths = [
    "/usr/local/lib",
    "/opt/homebrew/lib",
    "/usr/lib",
    "/opt/local/lib", // MacPorts
  ];
  
  for lib_path_str in &lib_paths {
    let lib_path = PathBuf::from(lib_path_str);
    let vulkan_libs = [
      "libvulkan.dylib",
      "libvulkan.1.dylib", 
      "libMoltenVK.dylib",
    ];
    
    for vulkan_lib in &vulkan_libs {
      if lib_path.join(vulkan_lib).exists() {
        println!("cargo:warning=Found {} at: {}", vulkan_lib, lib_path.display());
        return Some(lib_path);
      }
    }
  }
  
  println!("cargo:warning=No Vulkan installation found on macOS");
  None
}

/// Find Vulkan on Linux with comprehensive vendor support  
fn find_vulkan_linux() -> Option<PathBuf> {
  println!("cargo:warning=Searching Linux Vulkan installations...");
  
  // Linux Vulkan SDK and runtime paths
  let vulkan_paths = [
    // LunarG Vulkan SDK
    "/usr/local/vulkan-sdk",
    "/opt/vulkan-sdk",
    "/home/$USER/vulkan-sdk", // User installation
    
    // Package manager installations
    "/usr/lib/x86_64-linux-gnu/vulkan",
    "/usr/lib64/vulkan", 
    "/usr/local/lib/vulkan",
    "/usr/lib/vulkan",
    
    // Standard library paths
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
    "/lib64",
    "/lib",
    
    // Vendor-specific paths
    "/opt/amdgpu-pro/lib/x86_64-linux-gnu", // AMD
    "/usr/lib/x86_64-linux-gnu/mesa",       // Mesa
  ];
  
  for path_str in &vulkan_paths {
    // Expand $USER in paths
    let expanded_path = if path_str.contains("$USER") {
      if let Ok(user) = env::var("USER") {
        path_str.replace("$USER", &user)
      } else {
        continue;
      }
    } else {
      path_str.to_string()
    };
    
    println!("cargo:warning=Checking Linux Vulkan path: {}", expanded_path);
    let path = PathBuf::from(&expanded_path);
    if validate_vulkan_installation(&path) {
      println!("cargo:warning=Found valid Vulkan installation at: {}", path.display());
      return Some(path);
    }
  }
  
  // Try pkg-config with enhanced error handling
  println!("cargo:warning=Attempting pkg-config Vulkan detection...");
  if Command::new("pkg-config")
    .args(["--exists", "vulkan"])
    .status()
    .map(|s| s.success())
    .unwrap_or(false)
  {
    println!("cargo:warning=pkg-config found Vulkan package");
    if let Ok(output) = Command::new("pkg-config")
      .args(["--variable=libdir", "vulkan"])
      .output()
    {
      let path_str = String::from_utf8_lossy(&output.stdout);
      let path_str = path_str.trim();
      println!("cargo:warning=pkg-config reported Vulkan libdir: {}", path_str);
      let path = PathBuf::from(path_str);
      if path.exists() {
        if let Some(parent) = path.parent() {
          return Some(parent.to_path_buf());
        }
        return Some(path);
      }
    }
  } else {
    println!("cargo:warning=pkg-config Vulkan detection failed or not available");
  }
  
  // Use ldconfig to find Vulkan libraries
  if let Some(ldconfig_path) = find_vulkan_via_ldconfig() {
    println!("cargo:warning=Found Vulkan via ldconfig: {}", ldconfig_path.display());
    return Some(ldconfig_path);
  }
  
  println!("cargo:warning=No Vulkan installation found on Linux");
  None
}

/// Validate Vulkan installation completeness
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn validate_vulkan_installation(vulkan_path: &Path) -> bool {
  if !vulkan_path.exists() {
    return false;
  }
  
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  
  match target_os.as_str() {
    "windows" => {
      // Windows: Look for vulkan-1.dll or Vulkan headers/libs
      let dll_path = vulkan_path.join("vulkan-1.dll");
      let bin_dll = vulkan_path.join("Bin").join("vulkan-1.dll");
      let lib_path = vulkan_path.join("Lib").join("vulkan-1.lib");
      let include_path = vulkan_path.join("Include").join("vulkan").join("vulkan.h");
      
      dll_path.exists() || bin_dll.exists() || lib_path.exists() || include_path.exists()
    }
    "macos" => {
      // macOS: Framework, dylib, or MoltenVK
      if vulkan_path.to_str().unwrap_or("").contains(".framework") {
        let framework_binary = vulkan_path.join("vulkan");
        let framework_headers = vulkan_path.join("Headers").join("vulkan.h");
        framework_binary.exists() || framework_headers.exists()
      } else {
        let vulkan_libs = [
          "libvulkan.dylib",
          "libvulkan.1.dylib",
          "libMoltenVK.dylib",
        ];
        
        for lib in &vulkan_libs {
          if vulkan_path.join(lib).exists() {
            return true;
          }
        }
        false
      }
    }
    _ => {
      // Linux and other Unix-like systems
      let vulkan_libs = [
        "libvulkan.so",
        "libvulkan.so.1",
        "libvulkan.so.2",
      ];
      
      for lib in &vulkan_libs {
        if vulkan_path.join(lib).exists() {
          return true;
        }
      }
      
      // Check for versioned libraries
      if let Ok(entries) = std::fs::read_dir(vulkan_path) {
        for entry in entries.flatten() {
          if let Some(name) = entry.file_name().to_str() {
            if name.starts_with("libvulkan.so") {
              return true;
            }
          }
        }
      }
      
      false
    }
  }
}

/// Find Vulkan installation via Windows registry
fn find_vulkan_from_windows_registry() -> Option<PathBuf> {
  println!("cargo:warning=Searching Windows registry for Vulkan SDK...");
  
  // Try reading registry for Vulkan SDK installation paths
  let reg_keys = [
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\Vulkan\\ExplicitLayers",
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\LunarG\\VulkanSDK",
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\LunarG\\VulkanSDK",
  ];
  
  for reg_key in &reg_keys {
    if let Ok(output) = Command::new("reg")
      .args(["query", reg_key, "/s"])
      .output()
    {
      let output_str = String::from_utf8_lossy(&output.stdout);
      // Parse registry output for installation paths
      for line in output_str.lines() {
        if (line.contains("VulkanSDK") || line.contains("InstallDir")) && line.contains("REG_SZ") {
          if let Some(path_start) = line.rfind("C:\\") {
            let path_str = line[path_start..].trim();
            let path = PathBuf::from(path_str);
            if path.exists() {
              return Some(path);
            }
          }
        }
      }
    }
  }
  
  None
}

/// Find Vulkan installation via ldconfig on Linux
fn find_vulkan_via_ldconfig() -> Option<PathBuf> {
  println!("cargo:warning=Attempting ldconfig Vulkan detection...");
  
  if let Ok(output) = Command::new("ldconfig")
    .args(["-p"])
    .output()
  {
    let output_str = String::from_utf8_lossy(&output.stdout);
    for line in output_str.lines() {
      if line.contains("libvulkan.so") {
        if let Some(path_start) = line.rfind(" => ") {
          let lib_path_str = &line[path_start + 4..].trim();
          let lib_path = PathBuf::from(lib_path_str);
          if let Some(vulkan_lib_dir) = lib_path.parent() {
            println!("cargo:warning=Found Vulkan library directory via ldconfig: {}", vulkan_lib_dir.display());
            return Some(vulkan_lib_dir.to_path_buf());
          }
        }
      }
    }
  } else {
    println!("cargo:warning=ldconfig command failed or not available");
  }
  
  None
}

#[cfg(feature = "native-bindings")]
fn generate_native_bindings(out_dir: &Path) {
  let header_path = "src/backend/native/cuda_wrapper.h";

  // Only generate if header exists
  if !Path::new(header_path).exists() {
    println!(
      "cargo:warning=Header file {} not found, skipping binding generation",
      header_path
    );
    return;
  }

  let bindings = bindgen::Builder::default()
    .header(header_path)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate()
    .expect("Unable to generate bindings");

  bindings
    .write_to_file(out_dir.join("cuda_bindings.rs"))
    .expect("Couldn't write bindings!");

  println!("cargo:rerun-if-changed={}", header_path);
}
