/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 *
 *
 *   ✅ FP16 / BF16
 *   ✅ Causal mask
 *   ✅ Paged KV (page_table)
 *   ✅ MQA / GQA
 *   ✅ Varlen Q (cu_seqlens_q + max_seqlen_q)
 *   ❌ return_softmax_lse (lse always emitted; wrapper drops it on demand)
 *   ❌ SWA / window_size != (-1, -1)
 *   ❌ num_splits > 1 (FlashDecode)
 *   ❌ pack_gqa, leftpad_k
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <c10/core/Device.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "fai_host_api.hpp"
#include "fai_tiling.cpp"
#include "fai_tilingdata.h"
#include "fa_metadata_args.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_from_tensors.hpp"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using flash_attn_npu_950_v3::SeqlenScratch;
using flash_attn_npu_950_v3::fill_inference_context;

std::vector<at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        std::optional<at::Tensor> k_new_,
        std::optional<at::Tensor> v_new_,
        std::optional<at::Tensor> q_v_,
        std::optional<at::Tensor> out_,
        std::optional<at::Tensor> cu_seqlens_q_,
        std::optional<at::Tensor> cu_seqlens_k_,
        std::optional<at::Tensor> cu_seqlens_k_new_,
        std::optional<at::Tensor> seqused_q_,
        std::optional<at::Tensor> seqused_k_,
        std::optional<int64_t>    max_seqlen_q_,
        std::optional<int64_t>    max_seqlen_k_,
        std::optional<at::Tensor> page_table_,
        std::optional<at::Tensor> kv_batch_idx_,
        std::optional<at::Tensor> leftpad_k_,
        std::optional<at::Tensor> rotary_cos_,
        std::optional<at::Tensor> rotary_sin_,
        std::optional<at::Tensor> seqlens_rotary_,
        std::optional<at::Tensor> q_descale_,
        std::optional<at::Tensor> k_descale_,
        std::optional<at::Tensor> v_descale_,
        std::optional<float>      softmax_scale_,
        bool                      is_causal,
        int64_t                   window_size_left,
        int64_t                   window_size_right,
        int64_t                   attention_chunk,
        float                     softcap,
        bool                      is_rotary_interleaved,
        std::optional<at::Tensor> scheduler_metadata_,
        int64_t                   num_splits,
        std::optional<bool>       pack_gqa_,
        int64_t                   sm_margin)
{
    // ============================================================
    // 0. Device guard + stream + AIC core count
    // ============================================================
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // ============================================================
    // 1. dtype + stride sanity
    // ============================================================
    auto q_dtype = q.dtype();
    const bool is_bf16 = (q_dtype == torch::kBFloat16);
    const bool is_fp16 = (q_dtype == torch::kFloat16);
    TORCH_CHECK(is_bf16 || is_fp16,
                "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1,
                "query must be on NPU");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "query, key and value must be on the same NPU device");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);

    // ============================================================
    // 2. reject list
    // ============================================================
    TORCH_CHECK(!leftpad_k_.has_value(),
                "950 backend (v3) does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value()
                && !seqlens_rotary_.has_value(),
                "950 backend (v3) does not support rotary embedding");
    TORCH_CHECK(!q_descale_.has_value() && !k_descale_.has_value()
                && !v_descale_.has_value(),
                "950 backend (v3) does not support FP8 descales");
    TORCH_CHECK(softcap == 0.0f, "950 backend (v3) does not support softcap");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "950 backend (v3) does not support SWA");
    TORCH_CHECK(attention_chunk == 0,
                "950 backend (v3) does not support attention_chunk");
    TORCH_CHECK(num_splits == 0 || num_splits == 1,
                "950 backend (v3) only supports num_splits=0 or 1");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(),
                "950 backend (v3) does not support pack_gqa");

    // ============================================================
    // 3. paged / varlen mode + per-tensor checks
    // ============================================================
    const bool paged_KV    = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();

    at::Tensor cu_seqlens_q, cu_seqlens_k, page_table, seqlens_k;

    if (paged_KV) {
        page_table = page_table_.value();
        TORCH_CHECK(page_table.device() == q.device(),
                    "page_table must be on the same NPU device as query");
        TORCH_CHECK(page_table.dtype() == torch::kInt32,
                    "page_table must have dtype int32");
        CHECK_CONTIGUOUS(page_table);
    }
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.device().type() == at::kPrivateUse1,
                    "cu_seqlens_q must be on NPU");
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
                    "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(),
                    "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.device().type() == at::kPrivateUse1,
                    "cu_seqlens_k must be on NPU");
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
                    "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV,
                    "If cu_seqlens_k is passed in, paged table is not supported");
    }
    TORCH_CHECK(seqused_k_.has_value(),
                "950 backend (v3) requires seqused_k (per-batch KV seqlen) — the "
                "Python wrapper passes cache_seqlens through this argument");
    seqlens_k = seqused_k_.value();
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(seqlens_k.device().type() == at::kPrivateUse1,
                "seqused_k must be on NPU");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    TORCH_CHECK(seqlens_k.dim() == 1, "seqused_k must be rank 1");

    // ============================================================
    // 4. Shape extraction and output tensor
    // ============================================================
    const int64_t expected_q_dim = is_varlen_q ? 3 : 4;
    const int64_t expected_kv_dim = (is_varlen_q && !paged_KV) ? 3 : 4;
    TORCH_CHECK(q.dim() == expected_q_dim,
                "query must be rank ", expected_q_dim,
                is_varlen_q ? " in TND layout" : " in BSND layout");
    TORCH_CHECK(k.dim() == expected_kv_dim && v.dim() == expected_kv_dim,
                "key and value must both be rank ", expected_kv_dim);
    TORCH_CHECK(k.dim() == v.dim(), "key and value must have the same rank");
    if (is_varlen_q) {
        TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
                    "cu_seqlens_q must be a rank-1 tensor with at least two elements");
    }
    const auto sizes = q.sizes();
    int batch_size, seqlen_q, num_heads, head_size_q;
    if (is_varlen_q) {
        batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
        num_heads = static_cast<int>(sizes[1]);
        head_size_q = static_cast<int>(sizes[2]);
    } else {
        batch_size = static_cast<int>(sizes[0]);
        seqlen_q = static_cast<int>(sizes[1]);
        num_heads = static_cast<int>(sizes[2]);
        head_size_q = static_cast<int>(sizes[3]);
    }
    const int max_num_blocks_per_seq = !paged_KV ? 0 : static_cast<int>(page_table.size(1));
    const int num_blocks = !paged_KV ? 0 : static_cast<int>(k.size(0));
    const int page_block_size = !paged_KV ? 128 : static_cast<int>(k.size(1));
    // k is 3D TND for flash_attn_varlen_func, but 4D
    // (batch/num_blocks, seqlen/block, kv_heads, head) for flash_attn_with_kvcache.
    const int num_heads_k = static_cast<int>(k.dim() == 3 ? k.size(1) : k.size(2));
    const int head_size_v = static_cast<int>(v.size(-1));

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(seqlen_q > 0 && num_heads > 0,
                "query sequence length and head count must be positive");
    TORCH_CHECK(seqlens_k.numel() == batch_size,
                "seqused_k must contain one KV length per batch");
    TORCH_CHECK(k.size(-1) == head_size_q,
                "query and key must have the same head dimension");
    TORCH_CHECK(head_size_q >= 1 && head_size_q <= 256,
                "FlashAttention supports q/k head dimensions in [1, 256]");
    TORCH_CHECK(head_size_v >= 1 && head_size_v <= 256,
                "FlashAttention supports value head dimensions in [1, 256]");
    TORCH_CHECK(!(page_block_size != 128 && page_block_size != 256 && page_block_size != 512 && page_block_size != 1024),
                "FlashAttention only supports page_block_size dimension 128 or 256 or 512 or 1024");
    TORCH_CHECK(num_heads_k > 0, "key/value head count must be positive");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");
    if (!is_varlen_q && !paged_KV) {
        TORCH_CHECK(q.size(0) == k.size(0),
                    "query and key/value batch sizes must match in BSND layout");
    }
    if (paged_KV) {
        TORCH_CHECK(num_blocks > 0 && max_num_blocks_per_seq > 0,
                    "paged KV cache and page_table must contain at least one block");
        TORCH_CHECK(page_table.size(0) == batch_size,
                    "page_table batch dimension must match query batch size");
    }

    std::vector<int64_t> output_sizes(q.sizes().begin(), q.sizes().end());
    output_sizes.back() = head_size_v;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype,
                    "output must have the same dtype as inputs");
        TORCH_CHECK(out.device() == q.device(),
                    "output must be on the same device as query");
        TORCH_CHECK(out.sizes().vec() == output_sizes,
                    "output shape must match query except for the last dimension, which must match value");
        CHECK_CONTIGUOUS(out);
    } else {
        out = at::empty(output_sizes, q.options());
    }

    // ============================================================
    // 6/7. Tiling source: precomputed AICPU metadata or host tiling
    // ============================================================
    uint8_t *tilingDevice = nullptr;
    uint8_t *maskDevice = nullptr;
    uint8_t *metaBase = nullptr;
    uint64_t workSpaceSize = 0;
    SeqlenScratch scratch;
    optiling::FAInferContext ctx;
    at::Tensor tiling_dev;
    at::Tensor mask_npu_tensor;

    if (scheduler_metadata_.has_value()) {
        auto schedMd = scheduler_metadata_.value();
        TORCH_CHECK(schedMd.dtype() == at::kByte,
                    "scheduler_metadata must be a byte tensor");
        TORCH_CHECK(schedMd.is_contiguous(),
                    "scheduler_metadata must be contiguous");
        TORCH_CHECK(schedMd.device().type() == at::kPrivateUse1,
                    "scheduler_metadata must be an NPU tensor");
        TORCH_CHECK(static_cast<uint64_t>(schedMd.nbytes()) >=
                        fa_metadata::MetadataBytesWithKv(is_causal, batch_size),
                    "scheduler_metadata buffer is too small for this call's causal flag");
        metaBase = static_cast<uint8_t *>(schedMd.data_ptr());
        tilingDevice = metaBase + fa_metadata::TilingOffset(is_causal);
        maskDevice = is_causal ? metaBase : nullptr;
        workSpaceSize = fa_metadata::WorkSpaceSize(blockDim);
        if (workSpaceSize < fa_metadata::WS_FLOOR) {
            workSpaceSize = fa_metadata::WS_FLOOR;
        }
    } else {
        at::Tensor cu_seqlen_q_cpu;
        if (is_varlen_q) {
            cu_seqlen_q_cpu = cu_seqlens_q.to(at::Device(at::kCPU));
        }
        at::Tensor seqlens_k_cpu = seqlens_k.to(at::Device(at::kCPU));

        fill_inference_context(
            ctx, scratch,
            q, k, v,
            is_varlen_q ? &cu_seqlen_q_cpu : nullptr,
            &seqlens_k_cpu,
            paged_KV, page_block_size, num_blocks, max_num_blocks_per_seq,
            is_causal, is_varlen_q, is_bf16,
            batch_size, seqlen_q, num_heads, num_heads_k,
            head_size_q, head_size_v,
            softmax_scale_.value_or(1.0f / std::sqrt(static_cast<float>(head_size_q))),
            /* lse_flag= */ true,
            /* layout_str= */ is_varlen_q ? "TND" : "BSND");

        FAInferTilingData tilingData{};
        {
            optiling::FAInferTiling tiler(ctx);
            tiler.SetCoreNum(blockDim);
            tiler.DoTiling(tilingData);
        }
        if (tilingData.workSpaceSize < fa_metadata::WS_FLOOR) {
            tilingData.workSpaceSize = fa_metadata::WS_FLOOR;
        }
        workSpaceSize = tilingData.workSpaceSize;

        at::Tensor tiling_cpu = at::empty(
            {static_cast<int64_t>(sizeof(FAInferTilingData))},
            at::device(c10::kCPU).dtype(at::kByte));
        std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &tilingData,
                    sizeof(FAInferTilingData));
        tiling_dev = tiling_cpu.to(at::Device(at::kPrivateUse1));
        tilingDevice = static_cast<uint8_t *>(tiling_dev.data_ptr());

        if (is_causal) {
            at::Tensor mask_cpu_tensor =
                at::triu(at::ones({2048, 2048},
                                  at::device(c10::kCPU).dtype(at::kByte)), 1);
            mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
            maskDevice = static_cast<uint8_t *>(mask_npu_tensor.data_ptr());
        }
    }

    // ============================================================
    // 8. Allocate output-side buffers on NPU
    // ============================================================
    auto workspace = at::empty(
        {static_cast<int64_t>(workSpaceSize)},
        at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor softmaxlse;
    if (is_varlen_q) {
        // Match v3's varlen lse shape: {total_q, num_heads}
        softmaxlse = at::empty({sizes[0], num_heads},
                               at::device(at::kPrivateUse1).dtype(at::kFloat));
    } else {
        softmaxlse = at::empty({batch_size, seqlen_q, num_heads},
                               at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    // ============================================================
    // 10. Build kernelKey + launch via fai_host::LaunchFAI
    // ============================================================
    const Format fmt = is_varlen_q ? Format::TND : Format::BSND;
    const CacheMode cacheMode = paged_KV ? CacheMode::pagedCache
                                           : CacheMode::normalCache;
    const PageShape pageShape = paged_KV ? PageShape::BnBsND
                                           : PageShape::normalShape;
    const uint32_t maskTypeKey = is_causal ? 1u : 0u;
    const uint32_t innerPrec = 0u; // FP32 accum
    const std::string dataType = is_bf16 ? "bf16" : "half";
    const std::string cacheLayout = "nd"; // nd only

    const uint32_t kernelKey = fai_host::BuildKernelKey(
        dataType, cacheLayout, maskTypeKey, innerPrec,
        fmt, cacheMode, pageShape);

    // device pointers
    auto qDev = static_cast<uint8_t*>(q.data_ptr());
    auto kDev = static_cast<uint8_t*>(k.data_ptr());
    auto vDev = static_cast<uint8_t*>(v.data_ptr());
    auto oDev = static_cast<uint8_t*>(out.data_ptr());
    auto lseDev = static_cast<uint8_t*>(softmaxlse.data_ptr());
    auto wsDev = static_cast<uint8_t*>(workspace.data_ptr());
    auto tilDev = tilingDevice;

    const auto i64_npu = at::device(at::kPrivateUse1).dtype(at::kLong);
    at::Tensor q_seq_i64 = is_varlen_q
        ? cu_seqlens_q
        : at::empty({batch_size}, i64_npu);
    at::Tensor kv_seq_i64;
    uint8_t *kvSeqDev = nullptr;
    if (is_varlen_kv) {
        kv_seq_i64 = cu_seqlens_k;
        kvSeqDev = static_cast<uint8_t *>(kv_seq_i64.data_ptr());
    } else if (is_varlen_q && !paged_KV) {
        if (scheduler_metadata_.has_value()) {
            // AICPU precomputed the cumulative KV seqlen list in the metadata
            // buffer, so the metadata path needs no host D2H or device cumsum.
            kvSeqDev = metaBase + fa_metadata::KvSeqlenOffset(is_causal);
        } else {
            kv_seq_i64 = at::from_blob(
                const_cast<int32_t *>(ctx.kvSeqlenList), {batch_size + 1},
                at::dtype(torch::kInt32).device(torch::kCPU)).to(at::Device(at::kPrivateUse1));
            kvSeqDev = static_cast<uint8_t *>(kv_seq_i64.data_ptr());
        }
    } else {
        kv_seq_i64 = seqlens_k;
        kvSeqDev = static_cast<uint8_t *>(kv_seq_i64.data_ptr());
    }
    auto qSeqDev  = static_cast<uint8_t*>(q_seq_i64.data_ptr());
    auto blockTableDev = paged_KV
        ? static_cast<uint8_t*>(page_table.data_ptr())
        : nullptr;
    const bool enableDN =
        (!is_causal) && (head_size_q <= 256) && (head_size_v <= 256) && (innerPrec == 0u);

    aclError launchErr = ACL_SUCCESS;
    auto launch_fa_infer = [&]() -> int {
        launchErr = fai_host::LaunchFAI(
            kernelKey, enableDN,
            blockDim, aclStream,
            qDev, kDev, vDev, maskDevice, blockTableDev,
            oDev, lseDev, qSeqDev, kvSeqDev,
            wsDev, tilDev);
        return 0;
    };
    // RunOpApiV2 keeps the kernel launch inside a torch_npu op context so it can
    // be captured by NPUGraph; an explicit aclrtSynchronizeStream here would
    // fail graph capture with a device error.
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);
    TORCH_CHECK(launchErr == ACL_SUCCESS,
                "950 backend (v3): unsupported kernelKey=", kernelKey,
                " (no launcher registered for "
                "dtype=", dataType, " cacheLayout=", cacheLayout,
                " maskType=", maskTypeKey, " innerPrec=", innerPrec,
                " layout=", (fmt == Format::TND ? "TND" : "BSND"),
                " cacheMode=", (paged_KV ? "paged" : "normal"),
                ")");

    at::Tensor empty_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    at::Tensor empty_softmax_lse_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    return {out, softmaxlse, empty_accum, empty_softmax_lse_accum};
}
