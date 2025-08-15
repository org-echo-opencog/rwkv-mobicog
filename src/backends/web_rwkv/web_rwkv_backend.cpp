#include <fstream>
#include <filesystem>

#include "backend.h"
#include "web_rwkv_ffi.h"
#include "web_rwkv_backend.h"
#include "commondef.h"

namespace rwkvmobile {

int web_rwkv_backend::init(void * extra) {
    ::init((uint64_t)time(NULL));
    return RWKV_SUCCESS;
}

int web_rwkv_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    int ret = 0;
    if (model_path.find("prefab") != std::string::npos) {
        load_prefab(model_path.c_str());
    } else if (model_path.find("ABC") != std::string::npos 
        || model_path.find("abc") != std::string::npos
        || model_path.find("MIDI") != std::string::npos
        || model_path.find("midi") != std::string::npos) {
        load_with_rescale(model_path.c_str(), 0, 0, 0, 999);
    } else if (model_path.find("extended") != std::string::npos) {
        load_extended(model_path.c_str(), 0, 0, 999);
    } else {
        if (model_path.find("0.1B") != std::string::npos
        || model_path.find("0.4B") != std::string::npos
        || model_path.find("0.1b") != std::string::npos
        || model_path.find("0.4b") != std::string::npos) {
            load(model_path.c_str(), 0, 0, 0);
        } else {
            load(model_path.c_str(), 0, 999, 0);
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
    std::vector<uint16_t> ids = {(uint16_t)id};
    auto ret = infer_raw_last(ids.data(), ids.size());
    if (!ret.len || !ret.logits) {
        return RWKV_ERROR_EVAL;
    }
    logits_len_from_backend = ret.len;
    // if (ret.len != vocab_size) {
    //     return RWKV_ERROR_EVAL;
    // }
    logits = ret.logits;
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    std::vector<uint16_t> ids_u16(ids.begin(), ids.end());
    auto ret = infer_raw_last(ids_u16.data(), ids_u16.size());
    if (!ret.len || !ret.logits) {
        return RWKV_ERROR_EVAL;
    }
    logits_len_from_backend = ret.len;
    // if (ret.len != vocab_size) {
    //     return RWKV_ERROR_EVAL;
    // }
    logits = ret.logits;
    return RWKV_SUCCESS;
}

void web_rwkv_backend::free_logits_if_allocated(float *& logits) {
    if (logits) {
        ModelOutput output;
        output.len = (uintptr_t)logits_len_from_backend;
        output.logits = logits;
        ::free_raw(output);
        logits = nullptr;
    }
}

bool web_rwkv_backend::is_available() {
    // TODO: Detect this
    return true;
}

int web_rwkv_backend::clear_state() {
    ::clear_state();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::get_state(std::any &state) {
    struct StateRaw raw = ::get_state();
    if (!raw.len || !raw.state) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    state = raw;
    return RWKV_SUCCESS;
}

int web_rwkv_backend::set_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    struct StateRaw raw = std::any_cast<struct StateRaw>(state);
    if (!raw.len || !raw.state) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    ::set_state(raw);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::free_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    struct StateRaw raw = std::any_cast<struct StateRaw>(state);
    if (!raw.len || !raw.state) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    ::free_state(raw);
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