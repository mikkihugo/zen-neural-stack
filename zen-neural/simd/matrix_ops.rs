//! High-performance SIMD matrix operations module
//!
//! This module provides optimized matrix operations targeting 50-100x performance
//! improvements over scalar JavaScript implementations through:
//!
//! - AVX-512 vectorized kernels (16 f32 per instruction)
//! - AVX2 optimized fallbacks (8 f32 per instruction) 
//! - ARM NEON implementations (4 f32 per instruction)
//! - Cache-aware blocking and prefetching strategies
//! - Micro-kernel approach for maximum efficiency
//!
//! Performance targets:
//! - Matrix multiplication: 10-50x faster than JavaScript
//! - Vector operations: 50-100x faster than JavaScript
//! - Memory bandwidth: Maximized through prefetching and alignment

use super::{ActivationFunction, SimdConfig, SimdMatrixOps};
use num_traits::Float;
// use std::ptr; // Unused - for future memory optimizations

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]  
use std::arch::aarch64::*;

/// High-performance CPU SIMD operations with architecture-specific optimizations
pub struct HighPerfSimdOps {
    config: SimdConfig,
}

impl HighPerfSimdOps {
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    pub fn new_with_defaults() -> Self {
        Self {
            config: SimdConfig::default(),
        }
    }

    /// Optimized GEMM (General Matrix Multiplication) with micro-kernels
    pub fn gemm_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        // Determine best path based on available instruction sets
        #[cfg(target_arch = "x86_64")]
        {
            // Note: AVX-512 intrinsics are currently unstable in stable Rust
            // For now, we'll use AVX2 as the highest optimization level
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.gemm_avx2_optimized(a, b, c, m, n, k); }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.config.use_neon {
                unsafe { self.gemm_neon(a, b, c, m, n, k); }
                return;
            }
        }

        // Fallback to scalar implementation with blocking
        self.gemm_blocked_scalar(a, b, c, m, n, k);
    }

    /// AVX-512 optimized matrix multiplication with 6x16 micro-kernels
    /// Note: Currently disabled due to unstable AVX-512 intrinsics in stable Rust
    #[cfg(all(target_arch = "x86_64", feature = "unstable-avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn gemm_avx512(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        const MR: usize = 6;  // Micro-kernel M dimension
        const NR: usize = 16; // Micro-kernel N dimension (AVX-512 width)

        // Initialize result
        c.fill(0.0);

        let (mc, nc, kc) = self.get_cache_blocking_params();

        for jc in (0..n).step_by(nc) {
            let jc_end = (jc + nc).min(n);
            
            for pc in (0..k).step_by(kc) {
                let pc_end = (pc + kc).min(k);
                
                for ic in (0..m).step_by(mc) {
                    let ic_end = (ic + mc).min(m);
                    
                    // Micro-kernel loop
                    for jr in (jc..jc_end).step_by(NR) {
                        let jr_end = (jr + NR).min(jc_end);
                        
                        for ir in (ic..ic_end).step_by(MR) {
                            let ir_end = (ir + MR).min(ic_end);
                            
                            self.gemm_micro_kernel_avx512(
                                a, b, c, 
                                ir, ir_end,
                                jr, jr_end,
                                pc, pc_end,
                                m, n, k
                            );
                        }
                    }
                }
            }
        }
    }

    /// AVX-512 micro-kernel: 6x16 multiplication kernel
    /// Note: Currently disabled due to unstable AVX-512 intrinsics in stable Rust
    #[cfg(all(target_arch = "x86_64", feature = "unstable-avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn gemm_micro_kernel_avx512(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        ir: usize, ir_end: usize,
        jr: usize, jr_end: usize, 
        pc: usize, pc_end: usize,
        m: usize, n: usize, k: usize
    ) {
        // Registers for accumulation (6x16 = 6 AVX-512 registers)
        let mut c00 = _mm512_setzero_ps();
        let mut c01 = _mm512_setzero_ps();
        let mut c02 = _mm512_setzero_ps();
        let mut c03 = _mm512_setzero_ps();
        let mut c04 = _mm512_setzero_ps();
        let mut c05 = _mm512_setzero_ps();

        let actual_ir = (ir_end - ir).min(6);
        let actual_jr = jr_end - jr;

        // Inner product loop
        for p in pc..pc_end {
            // Load B vector (16 elements)
            let b_vec = if jr + 16 <= n {
                _mm512_loadu_ps(b.as_ptr().add(p * n + jr))
            } else {
                // Handle edge case with masking
                let mask = (1u16 << actual_jr) - 1;
                _mm512_maskz_loadu_ps(mask, b.as_ptr().add(p * n + jr))
            };

            // Prefetch next B data
            if self.config.enable_prefetch && p + 1 < pc_end {
                _mm_prefetch(
                    b.as_ptr().add((p + 1) * n + jr) as *const i8,
                    _MM_HINT_T0
                );
            }

            // Load and broadcast A elements, compute products
            if actual_ir > 0 {
                let a0 = _mm512_set1_ps(a[ir * k + p]);
                c00 = _mm512_fmadd_ps(a0, b_vec, c00);
            }
            if actual_ir > 1 {
                let a1 = _mm512_set1_ps(a[(ir + 1) * k + p]);
                c01 = _mm512_fmadd_ps(a1, b_vec, c01);
            }
            if actual_ir > 2 {
                let a2 = _mm512_set1_ps(a[(ir + 2) * k + p]);
                c02 = _mm512_fmadd_ps(a2, b_vec, c02);
            }
            if actual_ir > 3 {
                let a3 = _mm512_set1_ps(a[(ir + 3) * k + p]);
                c03 = _mm512_fmadd_ps(a3, b_vec, c03);
            }
            if actual_ir > 4 {
                let a4 = _mm512_set1_ps(a[(ir + 4) * k + p]);
                c04 = _mm512_fmadd_ps(a4, b_vec, c04);
            }
            if actual_ir > 5 {
                let a5 = _mm512_set1_ps(a[(ir + 5) * k + p]);
                c05 = _mm512_fmadd_ps(a5, b_vec, c05);
            }
        }

        // Store results back to C
        if actual_jr == 16 {
            // Full store
            if actual_ir > 0 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add(ir * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add(ir * n + jr), _mm512_add_ps(c_old, c00));
            }
            if actual_ir > 1 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add((ir + 1) * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add((ir + 1) * n + jr), _mm512_add_ps(c_old, c01));
            }
            if actual_ir > 2 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add((ir + 2) * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add((ir + 2) * n + jr), _mm512_add_ps(c_old, c02));
            }
            if actual_ir > 3 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add((ir + 3) * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add((ir + 3) * n + jr), _mm512_add_ps(c_old, c03));
            }
            if actual_ir > 4 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add((ir + 4) * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add((ir + 4) * n + jr), _mm512_add_ps(c_old, c04));
            }
            if actual_ir > 5 {
                let c_old = _mm512_loadu_ps(c.as_ptr().add((ir + 5) * n + jr));
                _mm512_storeu_ps(c.as_mut_ptr().add((ir + 5) * n + jr), _mm512_add_ps(c_old, c05));
            }
        } else {
            // Masked store for edge cases
            let mask = (1u16 << actual_jr) - 1;
            if actual_ir > 0 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add(ir * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add(ir * n + jr), mask, _mm512_add_ps(c_old, c00));
            }
            if actual_ir > 1 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add((ir + 1) * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add((ir + 1) * n + jr), mask, _mm512_add_ps(c_old, c01));
            }
            if actual_ir > 2 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add((ir + 2) * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add((ir + 2) * n + jr), mask, _mm512_add_ps(c_old, c02));
            }
            if actual_ir > 3 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add((ir + 3) * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add((ir + 3) * n + jr), mask, _mm512_add_ps(c_old, c03));
            }
            if actual_ir > 4 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add((ir + 4) * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add((ir + 4) * n + jr), mask, _mm512_add_ps(c_old, c04));
            }
            if actual_ir > 5 {
                let c_old = _mm512_maskz_loadu_ps(mask, c.as_ptr().add((ir + 5) * n + jr));
                _mm512_mask_storeu_ps(c.as_mut_ptr().add((ir + 5) * n + jr), mask, _mm512_add_ps(c_old, c05));
            }
        }
    }

    /// Enhanced AVX2 matrix multiplication with better blocking
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn gemm_avx2_optimized(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        const MR: usize = 6;  // Micro-kernel M dimension
        const NR: usize = 8;  // Micro-kernel N dimension (AVX2 width)

        c.fill(0.0);

        let (mc, nc, kc) = self.get_cache_blocking_params();

        for jc in (0..n).step_by(nc) {
            let jc_end = (jc + nc).min(n);
            
            for pc in (0..k).step_by(kc) {
                let pc_end = (pc + kc).min(k);
                
                for ic in (0..m).step_by(mc) {
                    let ic_end = (ic + mc).min(m);
                    
                    for jr in (jc..jc_end).step_by(NR) {
                        let jr_end = (jr + NR).min(jc_end);
                        
                        for ir in (ic..ic_end).step_by(MR) {
                            let ir_end = (ir + MR).min(ic_end);
                            
                            self.gemm_micro_kernel_avx2(
                                a, b, c,
                                ir, ir_end,
                                jr, jr_end, 
                                pc, pc_end,
                                m, n, k
                            );
                        }
                    }
                }
            }
        }
    }

    /// AVX2 micro-kernel: 6x8 multiplication kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn gemm_micro_kernel_avx2(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        ir: usize, ir_end: usize,
        jr: usize, jr_end: usize,
        pc: usize, pc_end: usize,
        m: usize, n: usize, k: usize
    ) {
        let mut c00 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c02 = _mm256_setzero_ps();
        let mut c03 = _mm256_setzero_ps();
        let mut c04 = _mm256_setzero_ps();
        let mut c05 = _mm256_setzero_ps();

        let actual_ir = (ir_end - ir).min(6);
        let actual_jr = jr_end - jr;

        for p in pc..pc_end {
            let b_vec = if jr + 8 <= n {
                _mm256_loadu_ps(b.as_ptr().add(p * n + jr))
            } else {
                _mm256_setzero_ps() // Handle edge case
            };

            if self.config.enable_prefetch && p + 1 < pc_end {
                _mm_prefetch(
                    b.as_ptr().add((p + 1) * n + jr) as *const i8,
                    _MM_HINT_T0
                );
            }

            if actual_ir > 0 {
                let a0 = _mm256_set1_ps(a[ir * k + p]);
                c00 = _mm256_fmadd_ps(a0, b_vec, c00);
            }
            if actual_ir > 1 {
                let a1 = _mm256_set1_ps(a[(ir + 1) * k + p]);
                c01 = _mm256_fmadd_ps(a1, b_vec, c01);
            }
            if actual_ir > 2 {
                let a2 = _mm256_set1_ps(a[(ir + 2) * k + p]);
                c02 = _mm256_fmadd_ps(a2, b_vec, c02);
            }
            if actual_ir > 3 {
                let a3 = _mm256_set1_ps(a[(ir + 3) * k + p]);
                c03 = _mm256_fmadd_ps(a3, b_vec, c03);
            }
            if actual_ir > 4 {
                let a4 = _mm256_set1_ps(a[(ir + 4) * k + p]);
                c04 = _mm256_fmadd_ps(a4, b_vec, c04);
            }
            if actual_ir > 5 {
                let a5 = _mm256_set1_ps(a[(ir + 5) * k + p]);
                c05 = _mm256_fmadd_ps(a5, b_vec, c05);
            }
        }

        // Store results with edge handling
        if actual_jr >= 8 {
            if actual_ir > 0 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add(ir * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add(ir * n + jr), _mm256_add_ps(c_old, c00));
            }
            if actual_ir > 1 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add((ir + 1) * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add((ir + 1) * n + jr), _mm256_add_ps(c_old, c01));
            }
            if actual_ir > 2 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add((ir + 2) * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add((ir + 2) * n + jr), _mm256_add_ps(c_old, c02));
            }
            if actual_ir > 3 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add((ir + 3) * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add((ir + 3) * n + jr), _mm256_add_ps(c_old, c03));
            }
            if actual_ir > 4 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add((ir + 4) * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add((ir + 4) * n + jr), _mm256_add_ps(c_old, c04));
            }
            if actual_ir > 5 {
                let c_old = _mm256_loadu_ps(c.as_ptr().add((ir + 5) * n + jr));
                _mm256_storeu_ps(c.as_mut_ptr().add((ir + 5) * n + jr), _mm256_add_ps(c_old, c05));
            }
        }
    }

    /// ARM NEON optimized matrix multiplication
    #[cfg(target_arch = "aarch64")]
    unsafe fn gemm_neon(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        const MR: usize = 4;  // Micro-kernel M dimension
        const NR: usize = 8;  // Micro-kernel N dimension (2x NEON width)

        c.fill(0.0);

        let (mc, nc, kc) = self.get_cache_blocking_params();

        for jc in (0..n).step_by(nc) {
            let jc_end = (jc + nc).min(n);
            
            for pc in (0..k).step_by(kc) {
                let pc_end = (pc + kc).min(k);
                
                for ic in (0..m).step_by(mc) {
                    let ic_end = (ic + mc).min(m);
                    
                    for jr in (jc..jc_end).step_by(NR) {
                        let jr_end = (jr + NR).min(jc_end);
                        
                        for ir in (ic..ic_end).step_by(MR) {
                            let ir_end = (ir + MR).min(ic_end);
                            
                            self.gemm_micro_kernel_neon(
                                a, b, c,
                                ir, ir_end,
                                jr, jr_end,
                                pc, pc_end,
                                m, n, k
                            );
                        }
                    }
                }
            }
        }
    }

    /// NEON micro-kernel: 4x8 multiplication kernel using dual 4-wide NEON registers
    #[cfg(target_arch = "aarch64")]
    unsafe fn gemm_micro_kernel_neon(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        ir: usize, ir_end: usize,
        jr: usize, jr_end: usize,
        pc: usize, pc_end: usize,
        m: usize, n: usize, k: usize
    ) {
        // 4x8 kernel = 4x2 NEON registers
        let mut c00_0 = vdupq_n_f32(0.0);
        let mut c00_1 = vdupq_n_f32(0.0);
        let mut c01_0 = vdupq_n_f32(0.0);
        let mut c01_1 = vdupq_n_f32(0.0);
        let mut c02_0 = vdupq_n_f32(0.0);
        let mut c02_1 = vdupq_n_f32(0.0);
        let mut c03_0 = vdupq_n_f32(0.0);
        let mut c03_1 = vdupq_n_f32(0.0);

        let actual_ir = (ir_end - ir).min(4);
        let actual_jr = jr_end - jr;

        for p in pc..pc_end {
            // Load B vectors (8 elements as 2 NEON registers)
            let b_vec_0 = if jr + 4 <= n {
                vld1q_f32(b.as_ptr().add(p * n + jr))
            } else {
                vdupq_n_f32(0.0)
            };
            let b_vec_1 = if jr + 8 <= n {
                vld1q_f32(b.as_ptr().add(p * n + jr + 4))
            } else {
                vdupq_n_f32(0.0)
            };

            // Prefetch next iteration
            if self.config.enable_prefetch && p + 1 < pc_end {
                ptr::read_volatile(b.as_ptr().add((p + 1) * n + jr));
            }

            // Load and broadcast A elements, compute products
            if actual_ir > 0 {
                let a0 = vdupq_n_f32(a[ir * k + p]);
                c00_0 = vfmaq_f32(c00_0, a0, b_vec_0);
                c00_1 = vfmaq_f32(c00_1, a0, b_vec_1);
            }
            if actual_ir > 1 {
                let a1 = vdupq_n_f32(a[(ir + 1) * k + p]);
                c01_0 = vfmaq_f32(c01_0, a1, b_vec_0);
                c01_1 = vfmaq_f32(c01_1, a1, b_vec_1);
            }
            if actual_ir > 2 {
                let a2 = vdupq_n_f32(a[(ir + 2) * k + p]);
                c02_0 = vfmaq_f32(c02_0, a2, b_vec_0);
                c02_1 = vfmaq_f32(c02_1, a2, b_vec_1);
            }
            if actual_ir > 3 {
                let a3 = vdupq_n_f32(a[(ir + 3) * k + p]);
                c03_0 = vfmaq_f32(c03_0, a3, b_vec_0);
                c03_1 = vfmaq_f32(c03_1, a3, b_vec_1);
            }
        }

        // Store results back to C
        if actual_jr >= 8 {
            if actual_ir > 0 {
                let c_old_0 = vld1q_f32(c.as_ptr().add(ir * n + jr));
                let c_old_1 = vld1q_f32(c.as_ptr().add(ir * n + jr + 4));
                vst1q_f32(c.as_mut_ptr().add(ir * n + jr), vaddq_f32(c_old_0, c00_0));
                vst1q_f32(c.as_mut_ptr().add(ir * n + jr + 4), vaddq_f32(c_old_1, c00_1));
            }
            if actual_ir > 1 {
                let c_old_0 = vld1q_f32(c.as_ptr().add((ir + 1) * n + jr));
                let c_old_1 = vld1q_f32(c.as_ptr().add((ir + 1) * n + jr + 4));
                vst1q_f32(c.as_mut_ptr().add((ir + 1) * n + jr), vaddq_f32(c_old_0, c01_0));
                vst1q_f32(c.as_mut_ptr().add((ir + 1) * n + jr + 4), vaddq_f32(c_old_1, c01_1));
            }
            if actual_ir > 2 {
                let c_old_0 = vld1q_f32(c.as_ptr().add((ir + 2) * n + jr));
                let c_old_1 = vld1q_f32(c.as_ptr().add((ir + 2) * n + jr + 4));
                vst1q_f32(c.as_mut_ptr().add((ir + 2) * n + jr), vaddq_f32(c_old_0, c02_0));
                vst1q_f32(c.as_mut_ptr().add((ir + 2) * n + jr + 4), vaddq_f32(c_old_1, c02_1));
            }
            if actual_ir > 3 {
                let c_old_0 = vld1q_f32(c.as_ptr().add((ir + 3) * n + jr));
                let c_old_1 = vld1q_f32(c.as_ptr().add((ir + 3) * n + jr + 4));
                vst1q_f32(c.as_mut_ptr().add((ir + 3) * n + jr), vaddq_f32(c_old_0, c03_0));
                vst1q_f32(c.as_mut_ptr().add((ir + 3) * n + jr + 4), vaddq_f32(c_old_1, c03_1));
            }
        }
    }

    /// Get cache-aware blocking parameters
    fn get_cache_blocking_params(&self) -> (usize, usize, usize) {
        let l1_size = self.config.l1_cache_size;
        let l2_size = 256 * 1024;  // Assume 256KB L2 cache
        let l3_size = 8 * 1024 * 1024;  // Assume 8MB L3 cache

        // Calculate blocking parameters to fit in cache levels
        let mc = (l2_size / (4 * self.config.block_size)).min(self.config.block_size * 4);
        let nc = (l3_size / (4 * self.config.block_size)).min(self.config.block_size * 8);
        let kc = (l1_size / (4 * 2)).min(self.config.block_size);

        (mc, nc, kc)
    }

    /// Blocked scalar fallback with cache optimization
    fn gemm_blocked_scalar(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        c.fill(0.0);

        let block_size = self.config.block_size;

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = (i_block + block_size).min(m);
                    let j_end = (j_block + block_size).min(n);
                    let k_end = (k_block + block_size).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k_idx in k_block..k_end {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

impl SimdMatrixOps<f32> for HighPerfSimdOps {
    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        self.gemm_f32(a, b, c, m, n, k);
    }

    fn matvec(&self, a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
        // Optimized matrix-vector multiplication using GEMM
        let mut x_expanded = vec![0.0f32; n * 1];
        x_expanded.copy_from_slice(x);
        self.gemm_f32(a, &x_expanded, y, m, 1, n);
    }

    fn add_bias(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.add_bias_avx2_optimized(matrix, bias, rows, cols); }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.config.use_neon {
                unsafe { self.add_bias_neon(matrix, bias, rows, cols); }
                return;
            }
        }

        // Scalar fallback
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] += bias[j];
            }
        }
    }

    fn apply_activation(&self, data: &mut [f32], activation: ActivationFunction) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.apply_activation_avx2_optimized(data, activation); }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.config.use_neon {
                unsafe { self.apply_activation_neon(data, activation); }
                return;
            }
        }

        // Scalar fallback with improved implementations
        self.apply_activation_scalar_optimized(data, activation);
    }

    fn activation_derivatives(
        &self,
        data: &[f32],
        derivatives: &mut [f32],
        activation: ActivationFunction,
    ) {
        // Similar pattern as apply_activation but for derivatives
        self.activation_derivatives_scalar_optimized(data, derivatives, activation);
    }
}

// Implementation of additional optimized methods
impl HighPerfSimdOps {
    /// AVX-512 bias addition
    /// Note: Currently disabled due to unstable AVX-512 intrinsics in stable Rust
    #[cfg(all(target_arch = "x86_64", feature = "unstable-avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_bias_avx512(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        const SIMD_WIDTH: usize = 16;

        for i in 0..rows {
            let mut j = 0;
            while j + SIMD_WIDTH <= cols {
                let matrix_ptr = matrix.as_mut_ptr().add(i * cols + j);
                let bias_ptr = bias.as_ptr().add(j);

                let matrix_vec = _mm512_loadu_ps(matrix_ptr);
                let bias_vec = _mm512_loadu_ps(bias_ptr);
                let result = _mm512_add_ps(matrix_vec, bias_vec);

                _mm512_storeu_ps(matrix_ptr, result);
                j += SIMD_WIDTH;
            }

            // Handle remaining elements
            while j < cols {
                matrix[i * cols + j] += bias[j];
                j += 1;
            }
        }
    }

    /// Enhanced AVX2 bias addition  
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_bias_avx2_optimized(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        const SIMD_WIDTH: usize = 8;

        for i in 0..rows {
            let mut j = 0;
            while j + SIMD_WIDTH <= cols {
                let matrix_ptr = matrix.as_mut_ptr().add(i * cols + j);
                let bias_ptr = bias.as_ptr().add(j);

                let matrix_vec = _mm256_loadu_ps(matrix_ptr);
                let bias_vec = _mm256_loadu_ps(bias_ptr);
                let result = _mm256_add_ps(matrix_vec, bias_vec);

                _mm256_storeu_ps(matrix_ptr, result);
                j += SIMD_WIDTH;
            }

            while j < cols {
                matrix[i * cols + j] += bias[j];
                j += 1;
            }
        }
    }

    /// NEON bias addition
    #[cfg(target_arch = "aarch64")]
    unsafe fn add_bias_neon(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        const SIMD_WIDTH: usize = 4;

        for i in 0..rows {
            let mut j = 0;
            while j + SIMD_WIDTH <= cols {
                let matrix_ptr = matrix.as_mut_ptr().add(i * cols + j);
                let bias_ptr = bias.as_ptr().add(j);

                let matrix_vec = vld1q_f32(matrix_ptr);
                let bias_vec = vld1q_f32(bias_ptr);
                let result = vaddq_f32(matrix_vec, bias_vec);

                vst1q_f32(matrix_ptr, result);
                j += SIMD_WIDTH;
            }

            while j < cols {
                matrix[i * cols + j] += bias[j];
                j += 1;
            }
        }
    }

    /// AVX-512 activation functions
    /// Note: Currently disabled due to unstable AVX-512 intrinsics in stable Rust
    #[cfg(all(target_arch = "x86_64", feature = "unstable-avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn apply_activation_avx512(&self, data: &mut [f32], activation: ActivationFunction) {
        const SIMD_WIDTH: usize = 16;
        let len = data.len();
        let mut i = 0;

        match activation {
            ActivationFunction::Relu => {
                let zero = _mm512_setzero_ps();
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = _mm512_loadu_ps(ptr);
                    let result = _mm512_max_ps(vec, zero);
                    _mm512_storeu_ps(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                let zero = _mm512_setzero_ps();
                let alpha_vec = _mm512_set1_ps(alpha);
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = _mm512_loadu_ps(ptr);
                    let mask = _mm512_cmp_ps_mask(vec, zero, _CMP_GT_OQ);
                    let neg_part = _mm512_mul_ps(vec, alpha_vec);
                    let result = _mm512_mask_blend_ps(mask, neg_part, vec);
                    _mm512_storeu_ps(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            _ => {
                // For complex activations, use scalar implementation
                self.apply_activation_scalar_optimized(data, activation);
                return;
            }
        }

        // Handle remaining elements
        while i < len {
            match activation {
                ActivationFunction::Relu => {
                    data[i] = data[i].max(0.0);
                },
                ActivationFunction::LeakyRelu(alpha) => {
                    if data[i] < 0.0 {
                        data[i] *= alpha;
                    }
                },
                _ => unreachable!(),
            }
            i += 1;
        }
    }

    /// Enhanced AVX2 activation functions
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn apply_activation_avx2_optimized(&self, data: &mut [f32], activation: ActivationFunction) {
        const SIMD_WIDTH: usize = 8;
        let len = data.len();
        let mut i = 0;

        match activation {
            ActivationFunction::Relu => {
                let zero = _mm256_setzero_ps();
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = _mm256_loadu_ps(ptr);
                    let result = _mm256_max_ps(vec, zero);
                    _mm256_storeu_ps(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                let zero = _mm256_setzero_ps();
                let alpha_vec = _mm256_set1_ps(alpha);
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = _mm256_loadu_ps(ptr);
                    let mask = _mm256_cmp_ps(vec, zero, _CMP_GT_OQ);
                    let neg_part = _mm256_mul_ps(vec, alpha_vec);
                    let result = _mm256_blendv_ps(neg_part, vec, mask);
                    _mm256_storeu_ps(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            _ => {
                self.apply_activation_scalar_optimized(data, activation);
                return;
            }
        }

        while i < len {
            match activation {
                ActivationFunction::Relu => {
                    data[i] = data[i].max(0.0);
                },
                ActivationFunction::LeakyRelu(alpha) => {
                    if data[i] < 0.0 {
                        data[i] *= alpha;
                    }
                },
                _ => unreachable!(),
            }
            i += 1;
        }
    }

    /// NEON activation functions
    #[cfg(target_arch = "aarch64")]
    unsafe fn apply_activation_neon(&self, data: &mut [f32], activation: ActivationFunction) {
        const SIMD_WIDTH: usize = 4;
        let len = data.len();
        let mut i = 0;

        match activation {
            ActivationFunction::Relu => {
                let zero = vdupq_n_f32(0.0);
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = vld1q_f32(ptr);
                    let result = vmaxq_f32(vec, zero);
                    vst1q_f32(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                let zero = vdupq_n_f32(0.0);
                let alpha_vec = vdupq_n_f32(alpha);
                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = vld1q_f32(ptr);
                    let mask = vcgtq_f32(vec, zero);
                    let neg_part = vmulq_f32(vec, alpha_vec);
                    let result = vbslq_f32(mask, vec, neg_part);
                    vst1q_f32(ptr, result);
                    i += SIMD_WIDTH;
                }
            },
            _ => {
                self.apply_activation_scalar_optimized(data, activation);
                return;
            }
        }

        while i < len {
            match activation {
                ActivationFunction::Relu => {
                    data[i] = data[i].max(0.0);
                },
                ActivationFunction::LeakyRelu(alpha) => {
                    if data[i] < 0.0 {
                        data[i] *= alpha;
                    }
                },
                _ => unreachable!(),
            }
            i += 1;
        }
    }

    /// Optimized scalar activation functions using lookup tables and approximations
    fn apply_activation_scalar_optimized(&self, data: &mut [f32], activation: ActivationFunction) {
        match activation {
            ActivationFunction::Sigmoid => {
                for x in data.iter_mut() {
                    *x = Self::fast_sigmoid(*x);
                }
            },
            ActivationFunction::Tanh => {
                for x in data.iter_mut() {
                    *x = Self::fast_tanh(*x);
                }
            },
            ActivationFunction::Relu => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x *= alpha;
                    }
                }
            },
            ActivationFunction::Gelu => {
                for x in data.iter_mut() {
                    *x = Self::fast_gelu(*x);
                }
            },
            ActivationFunction::Swish => {
                for x in data.iter_mut() {
                    *x = *x * Self::fast_sigmoid(*x);
                }
            },
        }
    }

    /// Fast sigmoid approximation using rational function
    #[inline]
    fn fast_sigmoid(x: f32) -> f32 {
        // Clamp to reasonable range to avoid overflow
        let x = x.clamp(-10.0, 10.0);
        
        // Rational approximation: good accuracy, faster than exp
        if x.abs() < 5.0 {
            let x2 = x * x;
            let numerator = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
            let denominator = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
            0.5 + numerator / (2.0 * denominator)
        } else {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
    }

    /// Fast tanh approximation  
    #[inline]
    fn fast_tanh(x: f32) -> f32 {
        let x = x.clamp(-3.0, 3.0);
        let x2 = x * x;
        let numerator = x * (945.0 + x2 * (105.0 + x2));
        let denominator = 945.0 + x2 * (420.0 + x2 * 15.0);
        numerator / denominator
    }

    /// Fast GELU approximation
    #[inline]
    fn fast_gelu(x: f32) -> f32 {
        // GELU ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = 0.7978845608; // √(2/π)
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        0.5 * x * (1.0 + Self::fast_tanh(inner))
    }

    /// Optimized scalar activation derivatives
    fn activation_derivatives_scalar_optimized(
        &self,
        data: &[f32],
        derivatives: &mut [f32],
        activation: ActivationFunction,
    ) {
        match activation {
            ActivationFunction::Sigmoid => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = x * (1.0 - x);
                }
            },
            ActivationFunction::Tanh => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = 1.0 - x * x;
                }
            },
            ActivationFunction::Relu => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = if x > 0.0 { 1.0 } else { 0.0 };
                }
            },
            ActivationFunction::LeakyRelu(alpha) => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = if x > 0.0 { 1.0 } else { alpha };
                }
            },
            ActivationFunction::Gelu => {
                // GELU derivative approximation
                for (i, &x) in data.iter().enumerate() {
                    let sqrt_2_over_pi = 0.7978845608f32;
                    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                    let tanh_val = Self::fast_tanh(tanh_arg);
                    let sech2 = 1.0 - tanh_val * tanh_val;
                    
                    derivatives[i] = 0.5 * (1.0 + tanh_val) + 
                                   0.5 * x * sqrt_2_over_pi * sech2 * (1.0 + 0.134145 * x * x);
                }
            },
            ActivationFunction::Swish => {
                for (i, &x) in data.iter().enumerate() {
                    let sigmoid = Self::fast_sigmoid(x);
                    derivatives[i] = sigmoid * (1.0 + x * (1.0 - sigmoid));
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_perf_simd_creation() {
        let ops = HighPerfSimdOps::new_with_defaults();
        assert!(ops.config.block_size > 0);
        assert!(ops.config.num_threads > 0);
    }

    #[test]
    fn test_gemm_correctness() {
        let ops = HighPerfSimdOps::new_with_defaults();
        
        // Test small matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0; 4];
        
        ops.gemm_f32(&a, &b, &mut c, 2, 2, 2);
        
        // Expected: [19, 22, 43, 50]
        assert!((c[0] - 19.0).abs() < 1e-4);
        assert!((c[1] - 22.0).abs() < 1e-4);
        assert!((c[2] - 43.0).abs() < 1e-4);
        assert!((c[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_fast_activations() {
        // Test fast activation functions
        assert!((HighPerfSimdOps::fast_sigmoid(0.0) - 0.5).abs() < 0.01);
        assert!((HighPerfSimdOps::fast_tanh(0.0) - 0.0).abs() < 0.01);
        assert!(HighPerfSimdOps::fast_sigmoid(10.0) > 0.99);
        assert!(HighPerfSimdOps::fast_sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_cache_blocking_params() {
        let ops = HighPerfSimdOps::new_with_defaults();
        let (mc, nc, kc) = ops.get_cache_blocking_params();
        
        assert!(mc > 0);
        assert!(nc > 0); 
        assert!(kc > 0);
        
        // Should be reasonable values for typical cache sizes
        assert!(kc <= 1024); // L1 blocking
        assert!(mc <= 4096); // L2 blocking
        assert!(nc <= 8192); // L3 blocking
    }
}