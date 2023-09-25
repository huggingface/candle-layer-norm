extern crate core;

mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, Layout, Result, Shape, Tensor};
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use half::{bf16, f16};

pub struct LayerNorm {
    pub epsilon: f32,
    pub is_rms_norm: bool
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

impl LayerNorm {
    pub fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: &candle::CudaStorage,
        r_l: &Layout,
        g: &candle::CudaStorage,
        g_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle::CudaStorage, candle::CudaStorage, Shape)> {
        let dev = x.device();

        let out_shape = x_l.shape().clone();

        let x = x.as_cuda_slice::<f16>()?;
        let r = r.as_cuda_slice::<f16>()?;
        let g = g.as_cuda_slice::<f16>()?;
        let b = b.as_cuda_slice::<f16>()?;

        let x = x.slice(x_l.start_offset()..);
        let r = r.slice(r_l.start_offset()..);
        let g = g.slice(g_l.start_offset()..);
        let b = b.slice(b_l.start_offset()..);

        let rows = x_l.dims()[0];
        let cols = x_l.dims()[1];

        if !(cols % 8 == 0 && cols <= 8192) {
           candle::bail!("hidden size must be % 8 and <= 8192")
        }

        let x_stride = x_l.stride();
        let r_stride = r_l.stride();
        let g_stride = g_l.stride();
        let b_stride = b_l.stride();

        let x_rank = x_stride.len();
        let r_rank = r_stride.len();
        let g_rank = g_stride.len();
        let b_rank = b_stride.len();

        if x_rank != 2 || r_rank != 2  {
            candle::bail!(
                "layer-norm expects input tensors of rank r (x: {x_rank}, r: {r_rank}"
            )
        }
        if x_stride[x_rank - 1] != 1 {
            candle::bail!("the last dim of x must be contiguous {x_stride:?}")
        }
        if r_stride[r_rank - 1] != 1 {
            candle::bail!("the last dim of r must be contiguous {r_stride:?}")
        }
        if g_stride[g_rank - 1] != 1 {
            candle::bail!("the last dim of g must be contiguous {g_stride:?}")
        }
        if b_stride[b_rank - 1] != 1 {
            candle::bail!("the last dim of b must be contiguous {b_stride:?}")
        }

        let cols_rounded = if cols <= 1536 {
            round_multiple(cols, 256)
        } else if cols <= 3072 {
            round_multiple(cols, 512)
        } else {
            round_multiple(cols, 1024)
        };

        let mu = unsafe { dev.alloc::<f32>(rows) }.w()?;
        let rsigma = unsafe { dev.alloc::<f32>(rows) }.w()?;

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        let dst_add = unsafe { dev.alloc::<f16>(elem_count) }.w()?;

        let is_rms_norm = if self.is_rms_norm { 1 } else { 0 };


        unsafe {
            let x_ptr = *x.device_ptr() as *const core::ffi::c_void;
            let r_ptr = *r.device_ptr() as *const core::ffi::c_void;
            let g_ptr = *g.device_ptr() as *const core::ffi::c_void;
            let b_ptr = *b.device_ptr() as *const core::ffi::c_void;
            let dst_add_ptr = *dst_add.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let mu_ptr = *mu.device_ptr() as *const core::ffi::c_void;
            let rsigma_ptr = *rsigma.device_ptr() as *const core::ffi::c_void;

            let multi_processors_count = dev.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap();

            ffi::run_ln(
                x_ptr,
                r_ptr,
                g_ptr,
                b_ptr,
                dst_add_ptr,
                dst_ptr,
                mu_ptr,
                rsigma_ptr,

                self.epsilon,

                // cols_rounded as u32,
                256 as u32,
                rows as u32,
                cols as u32,
                multi_processors_count,

                2,
                2,
                2,
                2,
                2,

                is_rms_norm,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        let dst_add = candle::CudaStorage::wrap_cuda_slice(dst_add, dev.clone());

        Ok((dst, dst_add, out_shape))
    }
}

// impl candle::CustomOp3 for LayerNorm {
//     fn name(&self) -> &'static str {
//         "flash-attn-varlen"
//     }
//
//     fn cpu_fwd(
//         &self,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//     ) -> Result<(CpuStorage, Shape)> {
//         candle::bail!("no cpu support for flash-attn")
//     }
//
//     fn cuda_fwd(
//         &self,
//         q: &candle::CudaStorage,
//         q_l: &Layout,
//         k: &candle::CudaStorage,
//         k_l: &Layout,
//         v: &candle::CudaStorage,
//         v_l: &Layout,
//     ) -> Result<(candle::CudaStorage, Shape)> {
//         match q.dtype() {
//             candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
//             candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
//             dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
//         }
//     }
// }
//
// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with variable-length batching.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
// /// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
// /// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
// /// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
// /// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
// /// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
// /// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
// ///
// /// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
// /// `seqlen_1 + seqlen_2`, etc.
// ///
// /// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
// pub fn flash_attn_varlen(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     seqlens_q: &Tensor,
//     seqlens_k: &Tensor,
//     max_seqlen_q: usize,
//     max_seqlen_k: usize,
//     softmax_scale: f32,
//     causal: bool,
// ) -> Result<Tensor> {
//     let op = LayerNorm {
//         softmax_scale,
//         causal,
//         max_seqlen_q,
//         max_seqlen_k,
//         seqlens_q: seqlens_q.clone(),
//         seqlens_k: seqlens_k.clone(),
//     };
//     q.apply_op3(k, v, op)
// }

#[cfg(test)]
mod tests {
    use candle::{Device, DType, Storage};
    use super::*;

    #[test]
    fn test_layer_norm () -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 256), &device)?.to_dtype(DType::F16)?;
        let (x, x_l) = x.storage_and_layout();
        let r = Tensor::randn(0., 1., (4, 256), &device)?.to_dtype(DType::F16)?;
        let (r, r_l) = r.storage_and_layout();

        let g = Tensor::randn(0., 1., 256, &device)?.to_dtype(DType::F16)?;
        let (g, g_l) = g.storage_and_layout();
        let b = Tensor::randn(0., 1., 256, &device)?.to_dtype(DType::F16)?;
        let (b, b_l) = b.storage_and_layout();

        let x = match &*x {
            Storage::Cpu(_) => candle::bail!("x must be a cuda tensor"),
            Storage::Cuda(x) => x
        };
        let r = match &*r {
            Storage::Cpu(_) => candle::bail!("r must be a cuda tensor"),
            Storage::Cuda(r) => r
        };
        let g = match &*g {
            Storage::Cpu(_) => candle::bail!("g must be a cuda tensor"),
            Storage::Cuda(g) => g
        };
        let b = match &*b {
            Storage::Cpu(_) => candle::bail!("b must be a cuda tensor"),
            Storage::Cuda(b) => b
        };


        let ln = LayerNorm { epsilon: 1e-12, is_rms_norm: false };
        let (r, add_r, s) = ln.cuda_fwd_t::<f16>(x, x_l, r, r_l, g, g_l, b, b_l)?;
        Ok(())
    }

}