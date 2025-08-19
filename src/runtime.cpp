#include "runtime.h"
#include "backend.h"
#include "logger.h"
#include <functional>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "rmpack.h"
#ifdef ENABLE_WEBRWKV
#include "web_rwkv_backend.h"
#endif

#ifdef ENABLE_NCNN
#include "ncnn_rwkv_backend.h"
#endif

#ifdef ENABLE_LLAMACPP
#include "llama_cpp_backend.h"
#endif

#ifdef ENABLE_QNN
#include "qnn_backend.h"
#endif

#ifdef ENABLE_MNN
#include "mnn_rwkv_backend.h"
#endif

#ifdef ENABLE_COREML
#include "coreml_rwkv_backend.h"
#endif

#if defined(ENABLE_VISION)
#include "multimodal/vision/vision_encoder.h"
#endif

#if defined(ENABLE_WHISPER)
#include "multimodal/whisper/whisper_encoder.h"
#endif

#if defined(ENABLE_TTS)
#include "frontend_utils.h"
#endif

#if defined(ENABLE_TTS) || defined(ENABLE_WHISPER)
#include "audio.h"
#endif

namespace rwkvmobile {

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_WEBRWKV:
            return "web-rwkv";
        case RWKV_BACKEND_NCNN:
            return "ncnn";
        case RWKV_BACKEND_LLAMACPP:
            return "llama.cpp";
        case RWKV_BACKEND_QNN:
            return "qnn";
        case RWKV_BACKEND_MNN:
            return "mnn";
        case RWKV_BACKEND_COREML:
            return "coreml";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "web-rwkv") {
        return RWKV_BACKEND_WEBRWKV;
    } else if (backend == "ncnn") {
        return RWKV_BACKEND_NCNN;
    } else if (backend == "llama.cpp") {
        return RWKV_BACKEND_LLAMACPP;
    } else if (backend == "qnn") {
        return RWKV_BACKEND_QNN;
    } else if (backend == "mnn") {
        return RWKV_BACKEND_MNN;
    } else if (backend == "coreml") {
        return RWKV_BACKEND_COREML;
    }
    return -1;
}

int runtime::load_model(std::string model_path, std::string backend_name, std::string tokenizer_path, void * extra) {
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        LOGE("Invalid backend name: %s\n", backend_name.c_str());
        return RWKV_ERROR_BACKEND;
    }

    auto model_instance = std::make_unique<ModelInstance>();
    if (model_instance == nullptr) {
        LOGE("Failed to allocate memory for model instance\n");
        return RWKV_ERROR_ALLOC;
    }

    // 1. Create and initialize backend
    if (backend_id == RWKV_BACKEND_WEBRWKV) {
#ifdef ENABLE_WEBRWKV
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new web_rwkv_backend,
            [](execution_provider *p) { delete (web_rwkv_backend*)p; });
#else
        LOGE("WebRWKV backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_NCNN) {
#ifdef ENABLE_NCNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new ncnn_rwkv_backend,
            [](execution_provider *p) { delete (ncnn_rwkv_backend*)p; });
#else
        LOGE("NCNN backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_LLAMACPP) {
#ifdef ENABLE_LLAMACPP
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new llama_cpp_backend,
            [](execution_provider *p) { delete (llama_cpp_backend*)p; });
#else
        LOGE("LLaMa.cpp backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_QNN) {
#ifdef ENABLE_QNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new qnn_backend,
            [](execution_provider *p) { delete (qnn_backend*)p; });
#else
        LOGE("QNN backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_MNN) {
#ifdef ENABLE_MNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mnn_rwkv_backend,
            [](execution_provider *p) { delete (mnn_rwkv_backend*)p; });
#else
        LOGE("MNN backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_COREML) {
#ifdef ENABLE_COREML
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new coreml_rwkv_backend,
            [](execution_provider *p) { delete (coreml_rwkv_backend*)p; });
#else
        LOGE("CoreML backend is not supported on this platform\n");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else {
        LOGE("Unsupported backend: %s\n", backend_name.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
    }

    int ret = model_instance->backend->init(extra);
    if (ret) {
        LOGE("Failed to initialize backend: %s, errno = %d\n", backend_name.c_str(), ret);
        return ret;
    }

    // 2. Load model
    ret = model_instance->backend->load_model(model_path);
    if (ret) {
        LOGE("Failed to load model from: %s, errno = %d\n", model_path.c_str(), ret);
        return ret;
    }
    LOGI("Loaded model from: %s as model_id = %d\n", model_path.c_str(), _next_model_id);
    LOGI("Model num_layers: %d, num_heads: %d, hidden_size: %d, vocab_size: %d\n",
         model_instance->backend->n_layers, model_instance->backend->num_heads, model_instance->backend->hidden_size, model_instance->backend->vocab_size);
    model_instance->backend->clear_state();

    // 3. Load tokenizer
    if (!tokenizer_path.empty()) {
        model_instance->tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
            [](tokenizer_base *p) { delete (trie_tokenizer*)p; });
        if (model_instance->tokenizer == nullptr) {
            return RWKV_ERROR_TOKENIZER;
        }
        ret = model_instance->tokenizer->load(tokenizer_path);
        if (ret) return ret;
    }

    // 4. Create sampler
    model_instance->sampler = std::make_unique<NucleusSampler>();
    if (model_instance->sampler == nullptr) {
        return RWKV_ERROR_SAMPLER;
    }

    // 5. Store the instance and return its ID
    int model_id = _next_model_id++;
    _models[model_id] = std::move(model_instance);
    _models[model_id]->model_path = model_path;
    _models[model_id]->backend_name = backend_name;
    _models[model_id]->tokenizer_path = tokenizer_path;

    return model_id;
}

int runtime::release_model(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _models.erase(model_id);
    return RWKV_SUCCESS;
}

#ifdef ENABLE_VISION
int runtime::load_vision_encoder(int model_id, std::string model_path, std::string adapter_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<VisionEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->LoadModel(model_path, adapter_path);
}

int runtime::release_vision_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_WHISPER
int runtime::load_whisper_encoder(int model_id, std::string model_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<WhisperEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->LoadModel(model_path, "");
}

int runtime::release_whisper_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

int runtime::get_available_backend_ids(std::vector<int> &backend_ids) {
    backend_ids = std::vector<int>();

#ifdef ENABLE_WEBRWKV
    // Snapdragon platform doesn't support WebRWKV backend yet
    if (_soc_detect.get_platform_type() != PLATFORM_SNAPDRAGON) {
        backend_ids.push_back(RWKV_BACKEND_WEBRWKV);
    }
#endif

#ifdef ENABLE_NCNN
    backend_ids.push_back(RWKV_BACKEND_NCNN);
#endif

#ifdef ENABLE_LLAMACPP
    backend_ids.push_back(RWKV_BACKEND_LLAMACPP);
#endif

#ifdef ENABLE_QNN
    if (_soc_detect.get_platform_type() == PLATFORM_SNAPDRAGON) {
        auto supported_soc_names = std::vector<std::string>{"SM8650", "SM8635", "SM8550", "SM8475"};
        if (std::find(supported_soc_names.begin(), supported_soc_names.end(), _soc_detect.get_soc_partname()) != supported_soc_names.end()) {
            backend_ids.push_back(RWKV_BACKEND_QNN);
        }
    }
#endif

#ifdef ENABLE_COREML
    backend_ids.push_back(RWKV_BACKEND_COREML);
#endif

    return RWKV_SUCCESS;
}

std::string runtime::get_available_backends_str() {
    std::vector<int> backend_ids;
    get_available_backend_ids(backend_ids);
    std::string ret = "";
    for (auto id : backend_ids) {
        ret += backend_enum_to_str(id) + ",";
    }
    return ret;
}

int runtime::load_initial_state(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (state_path.find(".rmpack") == std::string::npos) {
        LOGE("the specified state file is not a rmpack file\n");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    RMPack state_pack(state_path);
    int hidden_size_config = state_pack.getConfig()["hidden_size"];
    auto files = state_pack.getFiles();
    if (files.size() != model->backend->n_layers) {
        LOGE("state file has %d layers, but model has %d layers\n", (int)files.size(), model->backend->n_layers);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (hidden_size_config != model->backend->hidden_size) {
        LOGE("state file has hidden size %d, but model has hidden size %d\n", hidden_size_config, model->backend->hidden_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    size_t state_size = state_pack.getFileSize(files[0].filename);

    std::vector<std::vector<half_float::half>> states(model->backend->n_layers);
    for (int i = 0; i < model->backend->n_layers; i++) {
        states[i].resize(state_size / sizeof(half_float::half));
        auto data = state_pack.readFileToMemory(files[i].filename);
        memcpy(states[i].data(), data, state_size);
        state_pack.freeFileMemory(files[i].filename);
    }
    model->backend->load_raw_states(states);

    return RWKV_SUCCESS;
}

void runtime::clear_initial_state(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->backend->zero_state();
    model->backend->get_state(model->backend->state_head->state);
    model->backend->state_head->delete_after();
}

int runtime::eval_logits(int model_id, int id, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = model->backend->eval(id, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits(int model_id, std::vector<int> ids, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    int ret;
    for (; i + _prefill_chunk_size <= ids.size(); i += _prefill_chunk_size) {
        auto ids_chunk = std::vector<int>(ids.begin() + i, ids.begin() + i + _prefill_chunk_size);
        ret = model->backend->eval(ids_chunk, logits);
        if (ret != RWKV_SUCCESS) return ret;
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += _prefill_chunk_size;
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    if (i < ids.size()) {
        auto ids_left = std::vector<int>(ids.begin() + i, ids.end());
        ret = model->backend->eval(ids_left, logits);
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += ids_left.size();
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    _prefill_speed = ids.size() * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits_with_embeddings(int model_id, const float *embeddings, int n_tokens, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model->backend->eval_with_embeddings(embeddings, n_tokens, logits);
    auto end = std::chrono::high_resolution_clock::now();
    if (n_tokens > 1) {
        _prefill_speed = n_tokens * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    } else {
        _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return ret;
}

std::string runtime::apply_chat_template(int model_id, std::vector<std::string> inputs, bool enable_reasoning) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);

    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    std::string text = model->prompt;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 2 == 0) {
            auto user_text = inputs[i];
            user_text = replace_text(user_text, "\r\n", "\n");
            user_text = replace_text(user_text, "\n\n", "\n");

            if (!model->user_role.empty()) {
                text += model->bos_token + model->user_role + ": " + inputs[i] + model->eos_token;
            } else {
                text += inputs[i] + model->eos_token;
            }
        } else {
            if (i == inputs.size() - 1) {
                text += model->response_role + ": " + inputs[i];
            } else {
                text += model->response_role + ": " + inputs[i] + model->eos_token;
            }
        }
    }

    if (inputs.size() % 2 != 0) {
        text +=  model->response_role + ":";
        if (enable_reasoning) {
            text += " " + model->thinking_token;
        }
    }
    return text;
}

int runtime::chat(int model_id, std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;

    if (_prefilling_thread.joinable() && _prefilling_thread.get_id() != std::this_thread::get_id()) {
        LOGD("Found prefilling thread, joining\n");
        _prefilling_thread.join();
        LOGD("_prefilling_thread finished.\n");
    }

    auto input_text = apply_chat_template(model_id, inputs, enable_reasoning);
    LOGD("Applied chat template: \"%s\"\n", input_text.c_str());
    std::vector<int> text_ids = model->tokenizer->encode(input_text);
    std::string debug_msg = "text_ids: ";
    for (auto id : text_ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    float *logits = nullptr;
    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(text_ids, tokens_to_prefill);

    if (tokens_to_prefill.size() > 0) {
        _prefill_progress_start(tokens_to_prefill.size());
        std::string debug_msg = "new tokens to prefill: ";
        for (auto id : tokens_to_prefill) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGD("%s\n", debug_msg.c_str());

        // save a state checkpoint every about 256 tokens
        int checkpoint_interval = 256;
        for (int i = 0; i < tokens_to_prefill.size(); i += checkpoint_interval) {
            std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + i, tokens_to_prefill.begin() + std::min(i + checkpoint_interval, (int)tokens_to_prefill.size()));
            eval_logits(model_id, tokens_to_prefill_chunk, logits);
            int ret = model->backend->register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
            if (ret) return ret;
            model->backend->free_logits_if_allocated(logits);
        }
    }
    _prefill_progress_finish();

    model->response_buffer = input_text.substr(input_text.rfind(model->response_role + ":") + (model->response_role + ":").size());
    std::vector<int> response_ids_raw;
    model->response_buffer_ids = model->tokenizer->encode(model->response_buffer);
    int ret;

    model->sampler->clear_occurences();
    for (int i = 1; i < inputs.size(); i += 2) {
        std::vector<int> ids = model->tokenizer->encode(" " + inputs[i]);
        for (auto id: ids) {
            model->sampler->update_occurences(id);
        }
    }

    if (logits == nullptr) {
        if (node->last_logits.size() == model->backend->get_num_vocab()) {
            logits = node->last_logits.data();
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            ret = eval_logits(model_id, text_ids.back(), logits);
            if (ret) return ret;
            response_ids_raw.emplace_back(text_ids.back());
        }
    }

    int decoded_idx = 0;
    bool thinking_end_tag_found = false;
    bool is_pseudo_thinking = enable_reasoning && model->response_buffer.find("</think>") != std::string::npos;
    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());

        if (is_pseudo_thinking && i == 0) {
            // token 61 is '<', 261 is '\n\n'
            logits[61] = -1e9f;
            logits[261] = -1e9f;
        } else if (is_pseudo_thinking && i == 1 && decoded_idx == 11) {
            logits[61] = -1e9f;
        }

        decoded_idx = model->sampler->sample(logits, model->backend->get_num_vocab());
        if (decoded_idx == 0) {
            break;
        }

        std::string decoded = model->tokenizer->decode(decoded_idx);
        std::string tmp = model->response_buffer + decoded;
        for (auto &stop_code : model->stop_codes) {
            if (enable_reasoning && !thinking_end_tag_found && stop_code == "\n\n") {
                continue;
            }
            if (tmp.size() >= stop_code.size() &&
                tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                LOGD("stop code found: %s\n", stop_code.c_str());
                model->response_buffer_eos_found = true;
                break;
            }
        }

        if (enable_reasoning && !thinking_end_tag_found) {
            if (tmp.find("</think>") != std::string::npos) {
                thinking_end_tag_found = true;
            }
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", model->response_buffer_eos_found, model->stop_signal);
            break;
        }

        if (i != 0 || logits != node->last_logits.data()) {
            model->backend->free_logits_if_allocated(logits);
        }
        ret = eval_logits(model_id, decoded_idx, logits);
        if (ret) return ret;

        response_ids_raw.emplace_back(decoded_idx);
        model->response_buffer += decoded;
        model->response_buffer_ids.emplace_back(decoded_idx);
        if (i == 0 && model->response_buffer[0] == ' ') {
            model->response_buffer = model->response_buffer.substr(1);
        }

        model->sampler->update_occurences(decoded_idx);
        if (callback) {
            callback(model->response_buffer.c_str(), decoded_idx, decoded.c_str());
        }
    }

    if (response_ids_raw.size() > 0) {
        int ret = model->backend->register_state_checkpoint(node, response_ids_raw, logits);
        if (ret) return ret;
    }

    if (logits != node->last_logits.data()) {
        model->backend->free_logits_if_allocated(logits);
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::set_prompt(int model_id, std::string prompt) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    LOGD("Setting and processing prompt for model %d: \"%s\"\n", model_id, prompt.c_str());
    std::vector<int> ids = model->tokenizer->encode(prompt);
    if (model->backend->state_head->next == nullptr) {
        model->backend->state_head->next = new state_node;
        if (model->backend->state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (model->backend->state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    model->prompt = prompt;
    model->backend->set_state(model->backend->state_head->state);
    model->backend->state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (model->backend->state_head->next->state.has_value()) {
        model->backend->free_state(model->backend->state_head->next->state);
    }
    float *logits = nullptr;
    int ret = eval_logits(model_id, ids, logits);
    if (ret) {
        return ret;
    }
    model->backend->get_state(model->backend->state_head->next->state);
    model->backend->state_head->next->last_logits.resize(model->backend->get_num_vocab());
    memcpy(model->backend->state_head->next->last_logits.data(), logits, model->backend->get_num_vocab() * sizeof(float));
    model->backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}

std::string runtime::get_prompt(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->prompt;
}

#ifdef ENABLE_VISION
int runtime::set_image_prompt(int model_id, std::string path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr || model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<img src=\"" + path + "\">";
    std::vector<int> ids = model->tokenizer->encode(prompt);

    if (model->backend->state_head->next == nullptr) {
        model->backend->state_head->next = new state_node;
        if (model->backend->state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (model->backend->state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    model->prompt = prompt;
    model->backend->set_state(model->backend->state_head->state);
    model->backend->state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (model->backend->state_head->next->state.has_value()) {
        model->backend->free_state(model->backend->state_head->next->state);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> embeddings;
    int n_tokens;
    if (!model->multimodal_encoder->Encode(path, embeddings, n_tokens)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("siglip duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    float *logits = nullptr;

    start = std::chrono::high_resolution_clock::now();
    int ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
    if (ret) {
        return ret;
    }
    end = std::chrono::high_resolution_clock::now();
    LOGI("eval_logits_with_embeddings duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    model->backend->get_state(model->backend->state_head->next->state);
    model->backend->state_head->next->last_logits.resize(model->backend->get_num_vocab());
    memcpy(model->backend->state_head->next->last_logits.data(), logits, model->backend->get_num_vocab() * sizeof(float));
    model->backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_WHISPER
int runtime::set_audio_prompt(int model_id, std::string path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr || model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<audio src=\"" + path + "\">";
    std::vector<int> ids = model->tokenizer->encode(prompt);

    if (model->backend->state_head->next == nullptr) {
        model->backend->state_head->next = new state_node;
        if (model->backend->state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (model->backend->state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    model->prompt = prompt;
    model->backend->set_state(model->backend->state_head->state);
    model->backend->state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (model->backend->state_head->next->state.has_value()) {
        model->backend->free_state(model->backend->state_head->next->state);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> embeddings;
    int n_tokens;
    if (!model->multimodal_encoder->Encode(path, embeddings, n_tokens)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("whisper duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    float *logits = nullptr;

    int ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
    if (ret) {
        return ret;
    }
    model->backend->get_state(model->backend->state_head->next->state);
    model->backend->state_head->next->last_logits.resize(model->backend->get_num_vocab());
    memcpy(model->backend->state_head->next->last_logits.data(), logits, model->backend->get_num_vocab() * sizeof(float));
    model->backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
int runtime::sparktts_load_models(
    std::string wav2vec2_path,
    std::string bicodec_tokenizer_path,
    std::string bicodec_detokenizer_path
) {
    _sparktts = std::make_unique<sparktts>();
    if (!_sparktts->load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int runtime::sparktts_release_models() {
    _sparktts = nullptr;
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;

    wav_file wav;
    wav.load(prompt_audio_path);
    wav.resample(16000);

    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;
    _sparktts->tokenize_audio(wav.samples, global_tokens, semantic_tokens);
    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    std::string full_text = prompt_audio_text + tts_text;
    auto text_tokens = tokenizer_encode(model_id, full_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i] + global_token_offset);
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
    for (int i = 0; i < semantic_tokens.size(); i++) {
        input_tokens.push_back(semantic_tokens[i]);
    }

    std::vector<int> output_tokens;

    static const int tts_max_length = 3000;
    static const int tts_top_k = 50;
    static const float tts_top_p = 0.95;
    static const float tts_temperature = 1.0;
    static const int tts_eos_token = 8192;

    auto start = std::chrono::high_resolution_clock::now();

    clear_state(model_id);
    float *logits = nullptr;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (int i = 0; i < tts_max_length; i++) {
        int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
        model->backend->free_logits_if_allocated(logits);
        if (idx == tts_eos_token) {
            break;
        }

        output_tokens.push_back(idx);
        ret = eval_logits(model_id, idx, logits);
        if (ret || !logits) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] LLM inference time: %lf ms", duration);
    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_streaming(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    _tts_output_samples_buffer.clear();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;

    bool read_from_cache = _sparktts->get_global_and_semantic_tokens(prompt_audio_path, _cache_dir, global_tokens, semantic_tokens);
    if (semantic_tokens.empty() || global_tokens.empty()) {
        LOGE("[TTS] Failed to get global and/or semantic tokens");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            std::string full_text = prompt_audio_text + text;
            LOGI("[TTS] LLM input text: %s", full_text.c_str());
            auto text_tokens = tokenizer_encode(model_id, full_text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
            for (int i = 0; i < text_tokens.size(); i++) {
                input_tokens.push_back(text_tokens[i]);
            }
            input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
            for (int i = 0; i < global_tokens.size(); i++) {
                input_tokens.push_back(global_tokens[i] + global_token_offset);
            }
            input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
            for (int i = 0; i < semantic_tokens.size(); i++) {
                input_tokens.push_back(semantic_tokens[i]);
            }

            clear_state(model_id);
            float *logits = nullptr;
            int ret = eval_logits(model_id, input_tokens, logits);
            if (ret || !logits) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }
            logits[tts_eos_token] = -1e9;

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                model->backend->free_logits_if_allocated(logits);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                ret = eval_logits(model_id, idx, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        int actual_chunk_size = _sparktts->initial_chunk_size;
        std::vector<int> semantic_tokens_buf;
        _sparktts->resize_detokenizer_model(actual_chunk_size);
        int semantic_token_pos = 0;
        while (!generation_finished) {
            if (output_tokens.size() - semantic_token_pos >= actual_chunk_size) {
                std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.begin() + semantic_token_pos + actual_chunk_size);
                semantic_token_pos += actual_chunk_size;
                int buffered_size = semantic_tokens_buf.size();
                std::vector<int> current_semantic_tokens = semantic_tokens_buf;
                current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
                semantic_tokens_buf = std::vector<int>(current_semantic_tokens.begin() + (current_semantic_tokens.size() - _sparktts->overlap_size), current_semantic_tokens.end());
                auto new_samples = _sparktts->detokenize_audio(global_tokens, current_semantic_tokens);
                if (_tts_output_samples_buffer.empty()) {
                    ttfa = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total_start).count();
                    actual_chunk_size = _sparktts->chunk_size;
                    _sparktts->resize_detokenizer_model(actual_chunk_size);
                }
                _tts_output_samples_buffer.insert(_tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (output_tokens.size() - semantic_token_pos > 0) {
            std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.end());
            int buffered_size = semantic_tokens_buf.size();
            std::vector<int> current_semantic_tokens = semantic_tokens_buf;
            current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
            auto new_samples = _sparktts->detokenize_audio(global_tokens, current_semantic_tokens);
            _tts_output_samples_buffer.insert(_tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
        }
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration);
    LOGI("[TTS] Output audio length: %lf s", _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] RTF (%s): %lf", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] TTFA (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}
#endif

int runtime::clear_state(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    if (model->sampler != nullptr) {
        model->sampler->clear_occurences();
    }
    if (model->backend != nullptr) {
        model->backend->clear_state();
    }
    return RWKV_SUCCESS;
}

int runtime::gen_completion(int model_id, std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *)) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    model->response_buffer = "";
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;
    model->is_generating = true;
    model->stop_signal = false;

    std::vector<int> ids = model->tokenizer->encode(prompt);
    _prefill_progress_start(ids.size());

    float *logits = nullptr;
    int ret = eval_logits(model_id, ids, logits);
    if (ret || !logits) {
        model->is_generating = false;
        return ret;
    }
    _prefill_progress_finish();

    model->response_buffer = prompt;
    model->response_buffer_ids = ids;
    static int idx = 0;
    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());
        idx = model->sampler->sample(logits, model->backend->get_num_vocab());

        model->backend->free_logits_if_allocated(logits);
        model->response_buffer_eos_found = (idx == stop_code);

        std::string next = model->tokenizer->decode(idx);
        model->response_buffer += next;
        model->response_buffer_ids.push_back(idx);
        ret = eval_logits(model_id, idx, logits);
        if (callback) {
            callback(model->response_buffer.c_str(), idx, next.c_str());
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            break;
        }

        model->sampler->update_occurences(idx);
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

double runtime::get_avg_decode_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_decode_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_decode_speed < 0) {
        return 0.0;
    } else {
        return _decode_speed;
    }
}

double runtime::get_avg_prefill_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_prefill_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_prefill_speed < 0) {
        return 0.0;
    } else {
        return _prefill_speed;
    }
}

void runtime::set_sampler_params(int model_id, float temperature, int top_k, float top_p) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_temperature(temperature);
    model->sampler->set_top_k(top_k);
    model->sampler->set_top_p(top_p);
}

void runtime::set_is_generating(int model_id, bool is_generating) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->is_generating = is_generating;
}

void runtime::set_stop_signal(int model_id, bool stop_signal) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->stop_signal = stop_signal;
}

bool runtime::get_stop_signal(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->stop_signal;
}

bool runtime::is_generating(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->is_generating;
}

float runtime::get_temperature(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_temperature();
}

int runtime::get_top_k(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_k();
}

float runtime::get_top_p(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_p();
}

void runtime::set_penalty_params(int model_id, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_presence_penalty(presence_penalty);
    model->sampler->set_frequency_penalty(frequency_penalty);
    model->sampler->set_penalty_decay(penalty_decay);
}

float runtime::get_presence_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_presence_penalty();
}

float runtime::get_frequency_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_frequency_penalty();
}

float runtime::get_penalty_decay(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_penalty_decay();
}

void runtime::set_user_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->user_role = role;
}

void runtime::set_response_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_role = role;
}

void runtime::set_bos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->bos_token = token;
}

void runtime::set_eos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->eos_token = token;
    model->stop_codes.clear();
    model->stop_codes.push_back(token);
}

void runtime::set_thinking_token(int model_id, std::string thinking_token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->thinking_token = thinking_token;
}

std::vector<int> runtime::tokenizer_encode(int model_id, std::string text) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return {};
    }
    return model->tokenizer->encode(text);
}

std::string runtime::tokenizer_decode(int model_id, std::vector<int> ids) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(ids);
}

std::string runtime::tokenizer_decode(int model_id, int id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(id);
}

void runtime::clear_response_buffer(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;
}

void runtime::backend_set_extra_str(int model_id, std::string str) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return;
    }
    model->backend->extra_str = str;
}

int runtime::release() {
    _models.clear();
    if (_embedding) {
        _embedding->release();
        _embedding = nullptr;
    }
#ifdef ENABLE_TTS
    if (_sparktts) {
        _sparktts = nullptr;
    }
#endif
    return RWKV_SUCCESS;
}

void runtime::free_logits_if_allocated(int model_id, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return;
    }
    model->backend->free_logits_if_allocated(logits);
}

int runtime::get_vocab_size(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return 0;
    }
    return model->backend->get_num_vocab();
}

std::string runtime::get_response_buffer_content(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->response_buffer;
}

const std::vector<int32_t> runtime::get_response_buffer_ids(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_ids;
}

bool runtime::get_response_buffer_eos_found(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_eos_found;
}

double runtime::get_prefill_progress(int model_id) {
    return _prefill_progress;
}

void runtime::set_token_banned(int model_id, std::vector<int> token_banned) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->sampler == nullptr) {
        return;
    }
    model->sampler->set_token_banned(token_banned);
}

std::vector<int> runtime::get_loaded_model_ids() {
    std::vector<int> model_ids;
    for (const auto& pair : _models) {
        model_ids.push_back(pair.first);
    }
    return model_ids;
}

std::map<int, std::map<std::string, std::string>> runtime::get_loaded_models_info() {
    std::map<int, std::map<std::string, std::string>> models_info;

    for (const auto& pair : _models) {
        int model_id = pair.first;
        const auto& model = pair.second;

        std::map<std::string, std::string> model_info;
        model_info["model_path"] = model->model_path;
        model_info["backend_name"] = model->backend_name;
        model_info["tokenizer_path"] = model->tokenizer_path;
        model_info["user_role"] = model->user_role;
        model_info["response_role"] = model->response_role;
        model_info["bos_token"] = model->bos_token;
        model_info["eos_token"] = model->eos_token;
        model_info["thinking_token"] = model->thinking_token;
        model_info["is_generating"] = model->is_generating ? "true" : "false";
        model_info["vocab_size"] = model->backend ? std::to_string(model->backend->get_num_vocab()) : "0";

        models_info[model_id] = model_info;
    }

    return models_info;
}

std::string runtime::get_model_path_by_id(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->model_path;
}

} // namespace rwkvmobile
