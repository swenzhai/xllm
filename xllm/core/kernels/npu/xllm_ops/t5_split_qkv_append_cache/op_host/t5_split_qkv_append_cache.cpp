/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstring>

#include "common/t5_split_qkv_append_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

bool validate_shapes(const gert::Shape& qkv,
                     const gert::Shape& past_key,
                     const gert::Shape& past_value,
                     int64_t head_num,
                     int64_t head_dim,
                     uint32_t vector_cores,
                     T5SplitQkvAppendCacheTilingData& tiling) {
  if (qkv.GetDimNum() != 3 || past_key.GetDimNum() != 4 ||
      past_value.GetDimNum() != 4 || past_key != past_value) {
    return false;
  }
  const int64_t batch = past_key.GetDim(0);
  const int64_t length = past_key.GetDim(2);
  if (!t5qkv::make_tiling(
          batch, head_num, length, head_dim, vector_cores, tiling)) {
    return false;
  }
  return past_key.GetDim(1) == head_num && past_key.GetDim(3) == head_dim &&
         qkv.GetDim(0) == batch && qkv.GetDim(1) == 1 &&
         qkv.GetDim(2) == 3 * head_num * head_dim;
}

bool valid_desc(const gert::CompileTimeTensorDesc* desc, ge::DataType dtype) {
  return desc != nullptr && desc->GetDataType() == dtype &&
         desc->GetStorageFormat() == ge::FORMAT_ND &&
         desc->GetOriginFormat() == ge::FORMAT_ND;
}

ge::graphStatus infer_shape(gert::InferShapeContext* context) {
  const auto* qkv = context->GetInputShape(0);
  const auto* past_key = context->GetInputShape(1);
  const auto* past_value = context->GetInputShape(2);
  const auto* attrs = context->GetAttrs();
  auto* query = context->GetOutputShape(0);
  auto* key = context->GetOutputShape(1);
  auto* value = context->GetOutputShape(2);
  if (qkv == nullptr || past_key == nullptr || past_value == nullptr ||
      attrs == nullptr || query == nullptr || key == nullptr ||
      value == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto* head_num = attrs->GetAttrPointer<int64_t>(0);
  const auto* head_dim = attrs->GetAttrPointer<int64_t>(1);
  T5SplitQkvAppendCacheTilingData tiling{};
  if (head_num == nullptr || head_dim == nullptr ||
      !validate_shapes(*qkv,
                       *past_key,
                       *past_value,
                       *head_num,
                       *head_dim,
                       /*vector_cores=*/1,
                       tiling)) {
    return ge::GRAPH_FAILED;
  }
  *query = gert::Shape({tiling.batch, tiling.head_num, 1, tiling.head_dim});
  *key = gert::Shape({tiling.batch,
                      tiling.head_num,
                      static_cast<int64_t>(tiling.past_length) + 1,
                      tiling.head_dim});
  *value = *key;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus infer_dtype(gert::InferDataTypeContext* context) {
  const ge::DataType dtype = context->GetInputDataType(0);
  if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
      context->GetInputDataType(1) != dtype ||
      context->GetInputDataType(2) != dtype) {
    return ge::GRAPH_FAILED;
  }
  context->SetOutputDataType(0, dtype);
  context->SetOutputDataType(1, dtype);
  context->SetOutputDataType(2, dtype);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus tile(gert::TilingContext* context) {
  const auto* qkv = context->GetInputShape(0);
  const auto* past_key = context->GetInputShape(1);
  const auto* past_value = context->GetInputShape(2);
  const auto* attrs = context->GetAttrs();
  const auto* qkv_desc = context->GetInputDesc(0);
  if (qkv == nullptr || past_key == nullptr || past_value == nullptr ||
      attrs == nullptr || qkv_desc == nullptr ||
      context->GetPlatformInfo() == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const ge::DataType dtype = qkv_desc->GetDataType();
  if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (!valid_desc(context->GetInputDesc(index), dtype) ||
        !valid_desc(context->GetOutputDesc(index), dtype)) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto* head_num = attrs->GetAttrPointer<int64_t>(0);
  const auto* head_dim = attrs->GetAttrPointer<int64_t>(1);
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  T5SplitQkvAppendCacheTilingData tiling{};
  if (head_num == nullptr || head_dim == nullptr ||
      !validate_shapes(qkv->GetStorageShape(),
                       past_key->GetStorageShape(),
                       past_value->GetStorageShape(),
                       *head_num,
                       *head_dim,
                       platform.GetCoreNumAiv(),
                       tiling)) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape query_shape(
      {tiling.batch, tiling.head_num, 1, tiling.head_dim});
  const gert::Shape cache_shape({tiling.batch,
                                 tiling.head_num,
                                 static_cast<int64_t>(tiling.past_length) + 1,
                                 tiling.head_dim});
  const auto* query = context->GetOutputShape(0);
  const auto* key = context->GetOutputShape(1);
  const auto* value = context->GetOutputShape(2);
  if (query == nullptr || key == nullptr || value == nullptr ||
      query->GetStorageShape() != query_shape ||
      key->GetStorageShape() != cache_shape ||
      value->GetStorageShape() != cache_shape) {
    return ge::GRAPH_FAILED;
  }
  auto* raw_tiling = context->GetRawTilingData();
  auto* workspace = context->GetWorkspaceSizes(1);
  if (raw_tiling == nullptr || raw_tiling->GetCapacity() < sizeof(tiling) ||
      workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(raw_tiling->GetData(), &tiling, sizeof(tiling));
  raw_tiling->SetDataSize(sizeof(tiling));
  workspace[0] = 0;
  context->SetBlockDim(tiling.block_dim);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class T5SplitQkvAppendCache final : public OpDef {
 public:
  explicit T5SplitQkvAppendCache(const char* name) : OpDef(name) {
    for (const char* input : {"qkv", "pastKey", "pastValue"}) {
      Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    for (const char* output : {"query", "key", "value"}) {
      Output(output)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    Attr("headNum").AttrType(OPTIONAL).Int(1);
    Attr("headDim").AttrType(OPTIONAL).Int(1);
    SetInferShape(infer_shape).SetInferDataType(infer_dtype);
    AICore().SetTiling(tile);
    AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(T5SplitQkvAppendCache);

}  // namespace ops
