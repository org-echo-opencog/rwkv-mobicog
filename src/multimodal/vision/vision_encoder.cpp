#include "vision_encoder.h"
#include "llava.h"
#include "commondef.h"
#include "logger.h"
#include <vector>

namespace rwkvmobile {

VisionEncoder::VisionEncoder() : vision_encoder_ptr(nullptr, [](clip_ctx* p) { if (p) clip_free(p); }) {}

VisionEncoder::~VisionEncoder() = default;

int VisionEncoder::LoadModel(const std::string &model_path, const std::string &adapter_path) {
    auto adapter_path_cstr = adapter_path.empty() ? NULL : adapter_path.c_str();
    vision_encoder_ptr.reset(clip_model_load(model_path.c_str(), adapter_path_cstr, 0));
    if (vision_encoder_ptr == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

bool VisionEncoder::Encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens) {
    auto embd = llava_image_embed_make_with_filename(vision_encoder_ptr.get(), 4, path.c_str());
    if (embd == nullptr) {
        return false;
    }
    n_tokens = embd->n_image_pos;
    int embedding_dim = clip_n_mmproj_embd(vision_encoder_ptr.get());
    size_t embedding_size = (size_t)n_tokens * embedding_dim;
    embeddings.assign(embd->embed, embd->embed + embedding_size);
    llava_image_embed_free(embd);
    return true;
}

} // namespace rwkvmobile
