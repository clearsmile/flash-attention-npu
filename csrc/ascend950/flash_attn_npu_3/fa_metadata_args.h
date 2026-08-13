/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef CSRC_ASCEND950_FLASH_ATTN_NPU_3_FA_METADATA_ARGS_H
#define CSRC_ASCEND950_FLASH_ATTN_NPU_3_FA_METADATA_ARGS_H

#include <cstdint>
#include <type_traits>

#include "fai_tilingdata.h"

static_assert(std::is_trivially_copyable<FAInferTilingData>::value,
              "FAInferTilingData must be trivially copyable to cross the AICPU/device boundary");

namespace fa_metadata {
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = static_cast<uint64_t>(MASK_DIM) * MASK_DIM;

constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = static_cast<uint64_t>(128) * 512;
constexpr uint32_t SIZE_OF_16BIT = 2;
constexpr uint32_t SIZE_OF_32BIT = 4;
constexpr uint32_t PRELAUNCH_NUM = 3;
constexpr uint64_t WS_FLOOR = uint64_t(1024) * 1024 * 32 * 4;  // 128 MiB

inline uint64_t TilingOffset(bool causal)
{
    return causal ? MASK_BYTES : 0;
}

inline uint64_t MetadataBytes(bool causal)
{
    return TilingOffset(causal) + sizeof(FAInferTilingData);
}

inline uint64_t KvSeqlenOffset(bool causal)
{
    return MetadataBytes(causal);
}

inline uint64_t KvSeqlenBytes(uint32_t batch)
{
    return (static_cast<uint64_t>(batch) + 1) * sizeof(int32_t);
}

inline uint64_t MetadataBytesWithKv(bool causal, uint32_t batch)
{
    return KvSeqlenOffset(causal) + KvSeqlenBytes(batch);
}

inline uint64_t Mm1OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t SmOnlineOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_16BIT * PRELAUNCH_NUM;
}

inline uint64_t Mm2OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t UpdateOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t WorkSpaceSize(uint64_t blockDim)
{
    return Mm1OutSize(blockDim) + SmOnlineOutSize(blockDim) +
           Mm2OutSize(blockDim) + UpdateOutSize(blockDim);
}
}  // namespace fa_metadata

struct FAMetadataArgs {
    uint64_t cuSeqlensQAddr;
    uint64_t seqlensKAddr;
    uint64_t metaOutAddr;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numHeadsK;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t maxQSeqlen;
    uint32_t maskType;
    uint32_t blockDim;
    uint32_t isVarlen;
    uint32_t isVarlenKv;
    uint32_t pagedKV;
    float softmaxScale;
};

#endif
