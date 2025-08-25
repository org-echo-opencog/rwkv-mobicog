#include <fstream>
#include <filesystem>

#include "backend.h"
#include "web_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"
#include <memory>

namespace rwkvmobile {

int web_rwkv_backend::init(void * extra) {
    ::init((uint64_t)time(NULL));
    return RWKV_SUCCESS;
}

int web_rwkv_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    bool use_fp16 = true;
    if (model_path.find("respark") != std::string::npos) {
        use_fp16 = false;
    }

    int ret = 0;
    if (model_path.find("prefab") != std::string::npos) {
        load_prefab(model_path.c_str(), use_fp16);
    } else if (model_path.find("ABC") != std::string::npos
        || model_path.find("abc") != std::string::npos
        || model_path.find("MIDI") != std::string::npos
        || model_path.find("midi") != std::string::npos) {
        load_with_rescale(model_path.c_str(), 0, 0, 0, 999, use_fp16);
    } else if (model_path.find("extended") != std::string::npos) {
        load_extended(model_path.c_str(), 0, 0, 999, use_fp16);
    } else {
        if (model_path.find("0.1B") != std::string::npos
        || model_path.find("0.4B") != std::string::npos
        || model_path.find("0.1b") != std::string::npos
        || model_path.find("0.4b") != std::string::npos) {
            load(model_path.c_str(), 999, 0, 0, use_fp16);
        } else {
            load(model_path.c_str(), 0, 999, 0, use_fp16);
        }
    }

    struct ModelInfoOutput info = get_model_info();
    version = info.version;
    n_layers = info.num_layer;
    num_heads = info.num_head;
    hidden_size = info.num_emb;
    vocab_size = info.num_vocab;
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval(int id, float *& logits) {
    uint32_t id_u32 = (uint32_t)id;
    auto ret = infer_raw_last(&id_u32, 1);
    if (!ret.len || !ret.logits) {
        LOGE("web_rwkv_backend::eval: failed to infer_raw_last");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }
    memcpy(logits_buffer.data(), ret.logits, vocab_size * sizeof(float));
    logits = logits_buffer.data();

    ::free_raw(ret);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    std::vector<uint32_t> ids_u32(ids.begin(), ids.end());
    auto ret = infer_raw_last((const uint32_t *)ids_u32.data(), ids_u32.size());
    if (!ret.len || !ret.logits) {
        return RWKV_ERROR_EVAL;
    }
    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }
    memcpy(logits_buffer.data(), ret.logits, vocab_size * sizeof(float));
    logits = logits_buffer.data();

    ::free_raw(ret);
    return RWKV_SUCCESS;
}

void web_rwkv_backend::free_logits_if_allocated(float *& logits) {
    return;
}

bool web_rwkv_backend::is_available() {
    // TODO: Detect this
    return true;
}

int web_rwkv_backend::zero_state() {
    ::clear_state();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::get_state(std::any &state) {
    struct StateRaw raw = ::get_state();
    if (!raw.len || !raw.state) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    state = std::make_shared<web_rwkv_state>(raw);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::set_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    try {
        const auto& state_ptr = std::any_cast<const std::shared_ptr<web_rwkv_state>&>(state);
        if (!state_ptr) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }
        const StateRaw& raw = state_ptr->raw;
        if (!raw.len || !raw.state) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }
        ::set_state(raw);
        return RWKV_SUCCESS;
    } catch (const std::bad_any_cast& e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
}

int web_rwkv_backend::free_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    state.reset();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::release_model() {
    ::release();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile