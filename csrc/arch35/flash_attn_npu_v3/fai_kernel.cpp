/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "online_softmax.hpp"
#include "rescale_o.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "fai_tilingdata.h"

using namespace Catlass;
using namespace tla;

static constexpr uint32_t PRE_LAUNCH = 2;
static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
static constexpr uint32_t UB_S_BUF_STAGES = 2;

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    Format qFormat,
    Format kvFormat,
    CacheMode kvcacheType,
    PageShape kvcacheShape,
    MaskCategory maskCategory,
    CacheLayout cacheLayout,
    bool enableDN>
class FAIKernel950 {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = std::conditional_t<enableDN, typename BlockMmadQK::ElementB, typename BlockMmadQK::ElementA>;
    using ElementK = std::conditional_t<enableDN, typename BlockMmadQK::ElementA, typename BlockMmadQK::ElementB>;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;
    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;

    using LayoutQ = std::conditional_t<enableDN, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<enableDN, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutMask = layout::RowMajor;

    __aicore__ inline
    FAIKernel950() {}

    __aicore__ inline
    void operator()(FAIKernelParams const &params)
    {
        __gm__ FAInferTilingData *faiTilingData =
            reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        //tiling data

        batch_ = faiTilingData->batch;
        qHeads_ = faiTilingData->numHeads;
        kvHeads_ = faiTilingData->kvHeads;
        embed_ = faiTilingData->embeddingSize;
        embedV_ = faiTilingData->embeddingSizeV;
        firstBatchTaskNum_ = faiTilingData->firstBatchTaskNum;
        totalTaskNum_ = faiTilingData->totalTaskNum;
        scaleValue_ = faiTilingData->scaleValue;
        // base tile info
        qBaseTile_ = faiTilingData->qBaseTile;
        kvBaseTile_ = faiTilingData->kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = faiTilingData->actSeqAval;
        // aligned seqlen q & kv
        qSeqlenAligned_ = faiTilingData->qSeqlenAligned;
        kvSeqlenAligned_ = faiTilingData->kvSeqlenAligned;
        maxNumBlocksPerBatch_ = faiTilingData->maxNumBlocksPerBatch;
        blockSize_ = faiTilingData->blockSize;
        numBlocks_ = faiTilingData->numBlocks;
        qkL1TileM_ = faiTilingData->qkL1TileM;
        qkL1TileN_ = faiTilingData->qkL1TileN;
        qkL1TileKLeft_ = faiTilingData->qkL1TileKLeft;
        qkL1TileKRight_ = faiTilingData->qkL1TileKRight;
        pvL1TileM_ = faiTilingData->pvL1TileM;
        pvL1TileN_ = faiTilingData->pvL1TileN;
        pvL1TileKLeft_ = faiTilingData->pvL1TileKLeft;
        pvL1TileKRight_ = faiTilingData->pvL1TileKRight;
        qL1BufNum_ = faiTilingData->qL1BufNum;
        kL1BufNum_ = faiTilingData->kL1BufNum;
        vL1BufNum_ = faiTilingData->vL1BufNum;
        pL1BufNum_ = faiTilingData->pL1BufNum;
        globalWindowSize_ = faiTilingData->globalWindowSize;   // SWA left window
        localWindowSize_  = faiTilingData->localWindowSize;    // SWA right window

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        auto pvL1AddrStart_ = qkL1TileM_ * qkL1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
            qkL1TileKRight_ * qkL1TileN_ * kL1BufNum_ * sizeof(ElementK);
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * i);
        }
        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;

        uint32_t ubSRoundTile = 0;
        if constexpr (std::is_same_v<ElementS, half>) {
            for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
                ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
                ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                    rowNumPerSubCore * colNumPerSubCore * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES +
                    rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
            }
            ubSRoundTile = 16;
        } else {
            for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
                ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                    32768 * i);
            }

            for (uint32_t i = 0; i < UB_S_BUF_STAGES; i++) {
                ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                    4 * 32768 +
                    rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
            }
            ubSRoundTile = 128;
        }

        Gemm::Block::BlockMmadQKTileHelper qkL1TileHelper(qkL1TileM_, qkL1TileN_, qkL1TileKLeft_, qkL1TileKRight_,
            qL1BufNum_, kL1BufNum_);
        Gemm::Block::BlockMmadPVTileHelper pvL1TileHelper(pvL1TileM_, pvL1TileN_, pvL1TileKLeft_, pvL1TileKRight_,
            pL1BufNum_, vL1BufNum_);

        qkL0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
        qkL0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        pvL0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        pvL0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, qkL1TileHelper);
        BlockMmadPV blockMmadPV(resource, pvL1AddrStart_, pvL1TileHelper);
#endif

#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        int64_t strideQ = 0;
        int64_t strideO = 0;
        int64_t strideK = 0;
        int64_t strideV = 0;
        int64_t kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1); // used for TND_NZ

        strideQ = qHeads_ * embed_;
        if constexpr (cacheLayout == CacheLayout::nd) {
            strideK = kvHeads_ * embed_;
            strideV = kvHeads_ * embedV_;
        } else {
            strideK = 16;
            strideV = 16;
        }
        strideO = qHeads_ * embedV_;

        uint32_t embedVRound = RoundUp(embedV_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t blockBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        int64_t qSeqlen = faiTilingData->maxQSeqlen;
        int64_t kvSeqlen = gActualKvseqlen.GetValue(curBatch);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        if constexpr (qFormat == Format::TND) {
            qSeqlen = static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
        }
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                qBOffset += qSeqlen * strideQ;
                if constexpr (kvcacheType == CacheMode::normalCache) {
                    kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                    vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                } else {
                    blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch_); 
                }
                oBOffset += qSeqlen * strideO;
                
                qSeqlen = faiTilingData->maxQSeqlen;
                kvSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
                if constexpr (qFormat == Format::TND) {
                    qSeqlen = static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
                }
                if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
                    kvSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
                }
                curTotalTaskNum += (qSeqlen + qBaseTile_ - 1) / qBaseTile_ * qHeads_;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSTileIdx = taskIdxCurBatch / qHeads_;
            uint32_t qHeadIdx = taskIdxCurBatch - qSTileIdx * qHeads_;
            uint32_t kvHeadIdx = qHeadIdx / groupSize;
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = qSTileIdx * 128;
            gmOffsetQ = qBOffset + qSOffset * strideQ + qHeadIdx * embed_;
            gmOffsetO = oBOffset + qSOffset * strideO + qHeadIdx * embedV_;
            int64_t kvSOffset = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache && kvcacheShape == PageShape::BnNBsD) {
                strideK = embed_;
                strideV = embedV_;
                gmOffsetK = static_cast<uint64_t>(kvHeadIdx) * embed_ * blockSize_;
                gmOffsetV = static_cast<uint64_t>(kvHeadIdx) * embedV_ * blockSize_;
            } else if (cacheLayout == CacheLayout::nd) {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx * embed_);
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx * embedV_);
            } else {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx * embed_) * kvNumTokens;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx * embedV_) * kvNumTokens;
            }
            uint32_t qsBlockNum =  (qSeqlen + qBaseTile_ - 1) / qBaseTile_;
            uint32_t rowNum = qSTileIdx == qsBlockNum - 1 ? qSeqlen - (qsBlockNum - 1) * qBaseTile_ : qBaseTile_;
            uint32_t rowNumRound = 0;
            if constexpr (enableDN) {
                rowNumRound = RoundUp(rowNum, 32);
            } else {
                rowNumRound = RoundUp(rowNum, 16);
            }
            uint32_t kvSTileSizeAct = kvBaseTile_;

            uint32_t noSkipKvS = kvSeqlen;
            isEmptyWindow_ = false;   // 每个 Q tile 重置空窗口标记
            // SWA (Sliding Window Attention) 窗口边界变量 — 对齐 arch22 mha_fwd_kvcache.cpp:560-601
            uint32_t kvSStartIdx = 0;
            bool notPreMask = true;
            bool notNextMask = true;
            int32_t windowSizeLeftStartLen = 0;
            int32_t windowSizeLeftEndLen = 0;
            int32_t windowSizeRightStartLen = 0;
            int32_t windowSizeRightEndLen = 0;
            int32_t delStartRow = 0;
            int32_t delEndRow = static_cast<int32_t>(qSeqlen);
            int32_t leftPoint = 0;
            int32_t rightPoint = 0;
            int32_t swaMaskRowOffset = 0;

            if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                uint32_t diffS = kvSeqlen - qSeqlen;
                noSkipKvS = (qSTileIdx + 1U) * qBaseTile_ + diffS;
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);
            } else if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                int32_t winL = static_cast<int32_t>(globalWindowSize_);
                int32_t winR = static_cast<int32_t>(localWindowSize_);
                // 对齐 arch22 mha_fwd_kvcache.cpp:572,583: Q/K 底部对齐偏移量。
                // kvSeqlen 可能大于 qSeqlen（此时 diffS > 0，Q[i] 对齐 KV[i+diffS]），
                // 也可能小于（diffS < 0）。**不能钳制为 0**，否则窗口在 KV 空间错位。
                int32_t diffS = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen);
                swaMaskRowOffset = static_cast<int32_t>(qSOffset) + diffS;

                // 左侧窗口边界：跳过 window_left 范围外的 KV blocks
                // winL == INT32_MAX: sentinel，无左边界约束（由 tiling_from_tensors.hpp 将 -1 转换而来）
                if (winL == INT32_MAX) {
                    kvSStartIdx = 0;
                } else if (winL < 0 && winL * (-1) >= static_cast<int32_t>(qSeqlen)) {
                    kvSStartIdx = static_cast<uint32_t>(kvSeqlen) / kvBaseTile_ + 1;
                } else {
                    leftPoint = diffS - winL;
                    windowSizeLeftStartLen = static_cast<int32_t>(qSTileIdx * qBaseTile_) + leftPoint;
                    windowSizeLeftEndLen = windowSizeLeftStartLen + static_cast<int32_t>(rowNum);
                    kvSStartIdx = static_cast<uint32_t>(
                        AscendC::Std::max(0, windowSizeLeftStartLen) / static_cast<int32_t>(kvBaseTile_));
                    notPreMask = false;
                }

                // 右侧窗口截断
                if (winR == INT32_MAX) {
                    noSkipKvS = kvSeqlen;
                } else if (winR < 0 && winR * (-1) >= static_cast<int32_t>(kvSeqlen)) {
                    noSkipKvS = 0;
                } else {
                    rightPoint = diffS + winR;
                    windowSizeRightStartLen = static_cast<int32_t>(qSTileIdx * qBaseTile_) + rightPoint;
                    windowSizeRightEndLen = windowSizeRightStartLen + static_cast<int32_t>(rowNum);
                    // 保护 RoundUp 免受负参数影响: 当 windowSizeRightEndLen < 0 时，
                    // 整个右窗口在当前 KV tile 之前，截断为 0
                    int32_t safeRightEnd = AscendC::Std::max(0, windowSizeRightEndLen);
                    noSkipKvS = static_cast<uint32_t>(
                        AscendC::Std::min(static_cast<int32_t>(kvSeqlen),
                            RoundUp(safeRightEnd, static_cast<int32_t>(kvBaseTile_))));
                    // 右边界完全落在 tile 0 之前时，保持全量循环，
                    // 由 delEndRow 统一清零无效行（对齐 arch22 mha_fwd_kvcache.cpp:587）
                    if (noSkipKvS == 0) {
                        noSkipKvS = kvSeqlen;
                    }
                    notNextMask = false;
                }

                // 整 Q tile 内无有效窗口的行范围（对齐 arch22 mha_fwd_kvcache.cpp:594-598）
                if (winL != INT32_MAX && windowSizeLeftEndLen > static_cast<int32_t>(kvSeqlen)) {
                    delStartRow = static_cast<int32_t>(kvSeqlen) - leftPoint;
                }
                if (winR != INT32_MAX && windowSizeRightStartLen < 0) {
                    delEndRow = -rightPoint;
                }
            }

            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));

            // 空窗口早退（对齐 arch22 mha_fwd_kvcache.cpp:617-624）：
            // kvSLoopNum == 0 或 kvSStartIdx >= kvSLoopNum 时，当前 q tile 没有任何
            // 有效 KV，将 gO 对应行清零；LSE 保持 host 预置 +inf（950 内核不写 LSE）。
            //
            // 注意：不能直接用 continue 跳过 KV 循环，也不能让循环空转（kvSLoopNum=0）：
            // blockMmadQK/blockMmadPV 内部通过 SetCrossCoreSync<4,...> 为两个 AIV 设置
            // 跨核 flag（ID + V0_V1_FLAG_ID_OFFSET=16 → flag 16/17/21/22），这些 flag 在
            // 循环体执行时才被 Set，而 ReleaseSyncFlags<4,4,4>() 等待它们。空转循环不
            // 执行 matmul → 这些 flag 从未被 Set → Release 的 CrossCoreWaitFlag 死锁
            // → aicore timeout（err=507014，日志 data_type28）。
            //
            // 修复：将 kvSLoopNum 置 1（虚拟 KV tile），让 AIC/AIV 流水正常执行一次
            // QK→softmax→PV，从而 Set 全部跨核 flag。gO 已清零，虚拟 tile 的 PV 结果
            // 通过 isEmptyWindow_ 标志在 epilogueRescaleO 处被丢弃（不写入 gO）。
            //
            // 注意：虚拟 tile 必须**始终从 tile 0 开始**（kvSStartIdx=0），不能保留
            // 原 kvSStartIdx：虚拟 tile 只执行 1 次 QK/PV，若从非零 tile 开始，则
            // 本核只会 Set 奇/偶 tile 对应的 flag 子集（如 tile 7 → 1/17 与 6/22，
            // 不 Set 0/16 与 5/21），而 ReleaseSyncFlags<4,4,4>() 无条件等待全部
            // 四个 V1 flag（16/17/21/22）。当本核是唯一不 Set 某个 flag 的核时，
            // CrossCoreWaitFlag 永久等待 → aicore timeout（err=507015，日志
            // data_type22: win=(-128,864)，Q tile 7 空窗口被修复为从 tile 7 开始）。
            //
            // 从 tile 0 开始（kvSLoopNum=1, kvSStartIdx=0）时，虚拟 tile 的 flag
            // 模式为 qkReadyFlag=0/16、softmaxReadyFlag=2/18、pvReadyFlag=5/21，
            // 与正常 Q tile 0 完全相同 —— 本核自设 Release 所需的全部 V1 flag，
            // 不依赖其他核补设，与 data_type28（已验证不再崩溃）路径一致。
            if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                if (kvSLoopNum == 0 || kvSStartIdx >= kvSLoopNum) {
                    isEmptyWindow_ = true;
#ifdef __DAV_VEC__
                    auto zeroUb = resource.ubBuf.template GetBufferByByte<ElementO>(0);
                    AscendC::Duplicate(zeroUb, static_cast<ElementO>(0), embedVRound);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t row = 0; row < rowNum; ++row) {
                        AscendC::DataCopy(
                            gO[gmOffsetO + row * strideO], zeroUb,
                            AscendC::DataCopyParams(1, embedV_ / 16, 0, 0));
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
#endif
                    // 统一虚拟 tile：kvSLoopNum=1, kvSStartIdx=0, noSkipKvS=128
                    // （覆盖 kvSLoopNum==0 与 kvSStartIdx>=kvSLoopNum 两种情况）
                    kvSLoopNum = 1;
                    kvSStartIdx = 0;
                    noSkipKvS = kvBaseTile_;
                }
            }
#ifdef __DAV_CUBE__
            uint32_t qShapeCol = strideQ;
            uint32_t kShapeCol = strideK;
            uint32_t vShapeCol = strideV;
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
            auto gmQLayoutTlaDN = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            auto gmQTensorTlaDN = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTlaDN, Arch::PositionGM{});
            
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            if constexpr (enableDN) {
                blockMmadQK.loadQGM(gmQTensorTlaDN, actualBlockShapeQ);
            } else {
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
            }
            uint32_t kShapeRow = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache) {
                kShapeRow = numBlocks_ * blockSize_;
            } else {
                kShapeRow = kvSeqlen;
            }
            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeCol, kShapeRow);
            auto gmKLayoutTlaDN = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmKTensorTlaDN = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTlaDN, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = strideO;
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t kvSTileIdx = kvSStartIdx; kvSTileIdx < kvSLoopNum + PRE_LAUNCH; kvSTileIdx++) {
                if (kvSTileIdx < kvSLoopNum) {
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    int64_t stride = 64;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, ubSRoundTile));
                    auto ubSLayoutTlaDN = tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, 16), 64);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    auto ubSTensorTlaDN = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTlaDN, Arch::PositionUB{});
                    uint32_t qkReadyFlagId = ubSBufId;
                    Arch::CrossCoreFlag qkReadyFlag(qkReadyFlagId);
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdx - kvSStartIdx, qkL0ATotalStages_, pvL0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdx - kvSStartIdx, qkL0BTotalStages_, pvL0BTotalStages_, kvSLoopNum, true);
                    if constexpr (enableDN) {
                        blockMmadQK(
                            gmKTensorTlaDN, ubSTensorTlaDN,
                            gBlockTable[blockBOffset],
                            actualBlockShapeQK,
                            blockSize_,
                            kvSTileIdx, 0, kvHeads_,
                            kvNumTokens,
                            kvBaseTile_, 0, 0, 0,
                            qkReadyFlag,
                            prefixSumL0AStages, 
                            prefixSumL0BStages);
                    } else {
                        blockMmadQK(
                            gmKTensorTla, ubSTensorTla,
                            gBlockTable[blockBOffset],
                            actualBlockShapeQK,
                            blockSize_,
                            kvSTileIdx, 0, kvHeads_,
                            kvNumTokens,
                            kvBaseTile_, 0, 0, 0,
                            qkReadyFlag,
                            prefixSumL0AStages, 
                            prefixSumL0BStages);
                    }
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    uint32_t l1PBufId = kvSTileIdx % pL1BufNum_;
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, Catlass::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId],
                    l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                        auto gmMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(2048, 2048);
                        auto gmMaskTensorTla = tla::MakeTensor(gMask, gmMaskLayoutTla, Arch::PositionGM{});
                        uint32_t triUp = noSkipKvS - rowNum;
                        uint32_t triDown = noSkipKvS;
                        uint32_t kvSStartIdx = kvSTileIdx * kvBaseTile_;
                        uint32_t kvSEndIdx = kvSStartIdx + kvSTileSizeAct;
                        bool doTriUMask = triUp < kvSEndIdx - 1;
                        if (doTriUMask) {
                            epilogueOnlineSoftmax(
                                l1PTensorTla,
                                gmMaskTensorTla,
                                actualBlockShapeQK,
                                (kvSTileIdx == 0),
                                ubSBufId,
                                l1PBufId,
                                qkReadyFlag,
                                softmaxReadyFlag,
                                triUp,
                                triDown,
                                0, 0,
                                kvSStartIdx,
                                kvSEndIdx,
                                1);
                        } else {
                            epilogueOnlineSoftmax(
                                l1PTensorTla,
                                actualBlockShapeQK,
                                (kvSTileIdx == 0),
                                ubSBufId,
                                l1PBufId,
                                qkReadyFlag,
                                softmaxReadyFlag);
                        }
                    } else if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                        // SWA: preMask/nextMask 双边界 triu 复用 — 对齐 arch22 mha_fwd_kvcache.cpp:762-816
                        // 判定当前 KV tile 是否与窗口左/右边界相交
                        uint32_t kvSStart = kvSTileIdx * kvBaseTile_;
                        uint32_t kvSEnd   = kvSStart + kvSTileSizeAct;

                        bool doTriUPreMask = notPreMask ? false :
                            (windowSizeLeftStartLen >= static_cast<int32_t>(kvSStart) &&
                             windowSizeLeftStartLen <  static_cast<int32_t>(kvSEnd)) ||
                            (windowSizeLeftEndLen   >  static_cast<int32_t>(kvSStart) &&
                             windowSizeLeftEndLen   <= static_cast<int32_t>(kvSEnd)) ||
                            (windowSizeLeftStartLen <= static_cast<int32_t>(kvSStart) &&
                             windowSizeLeftEndLen   >= static_cast<int32_t>(kvSEnd));

                        bool doTriUNextMask = notNextMask ? false :
                            (windowSizeRightStartLen >= static_cast<int32_t>(kvSStart) &&
                             windowSizeRightStartLen <  static_cast<int32_t>(kvSEnd)) ||
                            (windowSizeRightEndLen   >  static_cast<int32_t>(kvSStart) &&
                             windowSizeRightEndLen   <= static_cast<int32_t>(kvSEnd)) ||
                            (windowSizeRightStartLen <= static_cast<int32_t>(kvSStart) &&
                             windowSizeRightEndLen   >= static_cast<int32_t>(kvSEnd));

                        if (doTriUPreMask || doTriUNextMask) {
                            if constexpr (std::is_same_v<ElementS, float>) {
                                epilogueOnlineSoftmax(
                                    l1PTensorTla, gMask,
                                    actualBlockShapeQK,
                                    (kvSTileIdx == kvSStartIdx),
                                    ubSBufId, l1PBufId,
                                    qkReadyFlag, softmaxReadyFlag,
                                    swaMaskRowOffset,
                                    static_cast<int32_t>(kvSStart));
                            } else {
                                auto gmMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(2048, 2048);
                                auto gmMaskTensorTla = tla::MakeTensor(gMask, gmMaskLayoutTla, Arch::PositionGM{});
                                if (doTriUNextMask && !doTriUPreMask) {
                                    uint32_t triUp = static_cast<uint32_t>(
                                        AscendC::Std::max(0, windowSizeRightStartLen));
                                    uint32_t triDown = static_cast<uint32_t>(
                                        AscendC::Std::max(0, windowSizeRightEndLen));
                                    uint32_t kvSIdx = kvSTileIdx * kvBaseTile_;
                                    uint32_t kvEIdx = kvSIdx + kvSTileSizeAct;
                                    epilogueOnlineSoftmax(
                                        l1PTensorTla, gmMaskTensorTla,
                                        actualBlockShapeQK,
                                        (kvSTileIdx == kvSStartIdx),
                                        ubSBufId, l1PBufId,
                                        qkReadyFlag, softmaxReadyFlag,
                                        triUp, triDown, 0, 0,
                                        kvSIdx, kvEIdx,
                                        1);
                                } else {
                                    epilogueOnlineSoftmax(
                                        l1PTensorTla, gmMaskTensorTla,
                                        actualBlockShapeQK,
                                        (kvSTileIdx == kvSStartIdx),
                                        ubSBufId, l1PBufId,
                                        qkReadyFlag, softmaxReadyFlag,
                                        static_cast<int32_t>(kvSStart),
                                        doTriUPreMask, doTriUNextMask,
                                        windowSizeLeftStartLen, windowSizeLeftEndLen,
                                        windowSizeRightStartLen, windowSizeRightEndLen);
                                }
                            }
                        } else {
                            // 窗口内部 tile: 完全不读 mask — 走快速路径 ⚡
                            epilogueOnlineSoftmax(
                                l1PTensorTla,
                                actualBlockShapeQK,
                                (kvSTileIdx == kvSStartIdx),
                                ubSBufId, l1PBufId,
                                qkReadyFlag, softmaxReadyFlag);
                        }
                    } else {
                        if constexpr (enableDN) {
                            epilogueOnlineSoftmax(
                                l1PTensorTla,
                                actualBlockShapeQK,
                                (kvSTileIdx == 0),
                                ubSBufId,
                                l1PBufId,
                                qkReadyFlag,
                                softmaxReadyFlag, 1);
                        } else {
                            epilogueOnlineSoftmax(
                                l1PTensorTla,
                                actualBlockShapeQK,
                                (kvSTileIdx == 0),
                                ubSBufId,
                                l1PBufId,
                                qkReadyFlag,
                                softmaxReadyFlag);
                        }
                    }
#endif
                }
                if (kvSTileIdx >= kvSStartIdx + PRE_LAUNCH) {
                    uint32_t kvSTileIdxNow = kvSTileIdx - PRE_LAUNCH;
                    if (kvSTileIdxNow == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdxNow * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapePV{rowNum, embedV_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvSTileIdxNow % UB_S_OTMP_BUF_STAGES;
                    uint32_t pvReadyFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvSTileIdxNow % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedVRound);
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor[ubOTmpBufId],
                        ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint64_t prefixSumL0AStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdxNow - kvSStartIdx, qkL0ATotalStages_, pvL0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossqkpvPrefixSumL0ABStages(
                        kvSTileIdxNow - kvSStartIdx, qkL0BTotalStages_, pvL0BTotalStages_, kvSLoopNum, false);
                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gBlockTable[blockBOffset],
                        actualBlockShapePV,
                        blockSize_,
                        kvSTileIdxNow, 0, kvHeads_,
                        kvNumTokens,
                        kvBaseTile_, 0, 0, 0,
                        softmaxReadyFlag, pvReadyFlag,
                        prefixSumL0AStages, 
                        prefixSumL0BStages);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint32_t curTileMod = kvSTileIdxNow % (PRE_LAUNCH + 1);
                    if (isEmptyWindow_) {
                        // 空窗口虚拟 tile：gO 已清零，丢弃 PV 结果。
                        // 仅消费 pvReadyFlag（与 epilogueRescaleO 相同的 Wait/Set 配对），
                        // 维持 AIC→AIV 流水同步，避免 flag 残留导致后续 Q tile 错乱。
                        // 注意：.template 关键字必须存在，因为 WaitCrossCoreSync/
                        // SetCrossCoreSync 是依赖名成员函数模板。
                        epilogueRescaleO.template WaitCrossCoreSync<4, PIPE_V>(pvReadyFlag);
                        epilogueRescaleO.template SetCrossCoreSync<4, PIPE_V>(pvReadyFlag);
                    } else if constexpr (enableDN) {
                        epilogueRescaleO(
                            gmOTensorTla, actualBlockShapePV,
                            curTileMod, kvSTileIdxNow,
                            (kvSTileIdxNow == kvSStartIdx),
                            (kvSTileIdxNow == kvSLoopNum - 1),
                            pvReadyFlag, 1);
                    } else {
                        epilogueRescaleO(
                            gmOTensorTla, actualBlockShapePV,
                            curTileMod, kvSTileIdxNow,
                            (kvSTileIdxNow == kvSStartIdx),
                            (kvSTileIdxNow == kvSLoopNum - 1),
                            pvReadyFlag, 0);
                    }
#endif
                }
            }

            if constexpr (maskCategory == MaskCategory::MASK_SWA) {
#ifdef __DAV_VEC__
                // 窗口外行清零：整 Q tile 空窗口已在上方清零，这里补齐 tile 内
                // 部分行无有效 KV（例如 win=(64,0) 的前 64 行）导致的 NaN/脏输出。
                int32_t rowStartLocal = AscendC::Std::max(static_cast<int32_t>(0),
                    delStartRow - static_cast<int32_t>(qSOffset));
                int32_t rowEndLocal = AscendC::Std::min(static_cast<int32_t>(rowNum),
                    delEndRow - static_cast<int32_t>(qSOffset));
                if ((delStartRow > 0 || delEndRow < static_cast<int32_t>(qSeqlen)) &&
                    rowStartLocal < rowEndLocal) {
                    auto zeroUb = resource.ubBuf.template GetBufferByByte<ElementO>(0);
                    AscendC::Duplicate(zeroUb, static_cast<ElementO>(0), embedVRound);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (int32_t row = rowStartLocal; row < rowEndLocal; ++row) {
                        AscendC::DataCopy(
                            gO[gmOffsetO + row * strideO], zeroUb,
                            AscendC::DataCopyParams(1, embedV_ / 16, 0, 0));
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
#endif
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // cross core sync
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // mask
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(1);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline
    uint64_t CalcCrossqkpvPrefixSumL0ABStages(
        uint32_t KvSTileIdx, uint32_t singleqkL0Stages,
        uint32_t singlepvL0Stages, uint32_t kvSLoopNum,
        bool isCurPhaseqk)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseqk) {
            prefixSumStages = (KvSTileIdx <= PRE_LAUNCH) ?
                KvSTileIdx * singleqkL0Stages :
                KvSTileIdx * singleqkL0Stages + (KvSTileIdx - PRE_LAUNCH) * singlepvL0Stages;
        } else {
            // 修复 uint32_t 下溢：当 kvSLoopNum <= PRE_LAUNCH 时，
            // kvSLoopNum - PRE_LAUNCH 作为无符号数下溢为 0xFFFFFFFF，
            // 导致条件 KvSTileIdx < 0xFFFFFFFF 恒为 true，取错分支。
            // 当 kvSLoopNum <= PRE_LAUNCH 时，所有 QK tile 均已完成，
            // 直接使用 kvSLoopNum * sqk + KvSTileIdx * spv。
            if (kvSLoopNum > PRE_LAUNCH) {
                prefixSumStages = (KvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                    (KvSTileIdx + 1 + PRE_LAUNCH) * singleqkL0Stages + KvSTileIdx * singlepvL0Stages :
                    kvSLoopNum * singleqkL0Stages + KvSTileIdx * singlepvL0Stages;
            } else {
                prefixSumStages = kvSLoopNum * singleqkL0Stages + KvSTileIdx * singlepvL0Stages;
            }
        }
        return prefixSumStages;
    }

private:
    Arch::Resource<ArchTag> resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t embedV_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t maxNumBlocksPerBatch_;
    uint32_t blockSize_;
    uint32_t numBlocks_;

    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint32_t actSeqAval_;

    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;

    uint32_t qkL1TileM_;
    uint32_t qkL1TileN_;
    uint32_t qkL1TileKLeft_;
    uint32_t qkL1TileKRight_;
    uint32_t pvL1TileM_;
    uint32_t pvL1TileN_;
    uint32_t pvL1TileKLeft_;
    uint32_t pvL1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;

    int32_t globalWindowSize_;    // SWA: left window extent (int32 支持负数语义)
    int32_t localWindowSize_;     // SWA: right window extent
    bool isEmptyWindow_ = false;  // SWA 空窗口标记：丢弃虚拟 KV tile 的 PV 结果，保持 gO 清零

    uint32_t qkL0ATotalStages_;
    uint32_t qkL0BTotalStages_;
    uint32_t pvL0ATotalStages_;
    uint32_t pvL0BTotalStages_;
};

template <class InDtype, class SMDtype, 
        Format qFormat, Format kvFormat, 
        CacheMode kvcacheType, PageShape kvcacheShape, 
        MaskCategory maskCategory, CacheLayout cacheLayout>
CATLASS_GLOBAL void FAInfer(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<Q_BLK>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyQK = std::conditional_t<std::is_same_v<ElementS, half>, 
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>,
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;

   // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmax;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using FAIKernel950 = FAIKernel950<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout, false>;
    FAIKernelParams params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAIKernel950 faInfer;
    faInfer(params);
}

template <class InDtype, class SMDtype, 
        Format qFormat, Format kvFormat, 
        CacheMode kvcacheType, PageShape kvcacheShape, 
        MaskCategory maskCategory, CacheLayout cacheLayout>
CATLASS_GLOBAL void FAInferDn(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutK = layout::RowMajor;
    using LayoutQ = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<Q_BLK>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyQK = std::conditional_t<std::is_same_v<ElementS, half>, 
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>,
                        Gemm::Tile::PackedTileCopyTlaToUB<
                                    ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS, LayoutS,
                                    void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ, ElementS, void, TileCopyQK>;

   // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmax;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using FAIKernel950 = FAIKernel950<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout, true>;
    FAIKernelParams params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAIKernel950 faInfer;
    faInfer(params);
}
