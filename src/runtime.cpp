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

#ifdef ENABLE_VISION
#include "llava.h"
#include "clip.h"
#endif

#ifdef ENABLE_WHISPER
#include "whisper.h"
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

void runtime::apply_logits_penalties(float * logits, int vocab_size) {
    if (!logits) {
        return;
    }
    for (auto &[id, occurence] : _occurences) {
        if (id >= vocab_size) {
            continue;
        }
        logits[id] -=
            _frequency_penalty * occurence + _presence_penalty;
        _occurences[id] *= _penalty_decay;
    }

    for (auto &token_banned : _token_banned) {
        if (token_banned >= vocab_size) {
            continue;
        }
        logits[token_banned] = -INFINITY;
    }
}

int runtime::init(std::string backend_name) {
    return init(backend_name, nullptr);
}

int runtime::init(std::string backend_name, void * extra) {
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        return RWKV_ERROR_BACKEND;
    }
    int ret = init(backend_id, extra);
    if (!ret) {
        LOGI("Initialized runtime with backend: %s\n", backend_name.c_str());
    } else {
        LOGE("Failed to initialize runtime with backend: %s, errno = %d\n", backend_name.c_str(), ret);
    }
    return ret;
}

int runtime::init(int backend_id) {
    return init(backend_id, nullptr);
}

int runtime::init(int backend_id, void * extra) {
    _sampler = std::unique_ptr<sampler>(new sampler);
    if (_sampler == nullptr) {
        return RWKV_ERROR_SAMPLER;
    }

    if (backend_id == RWKV_BACKEND_WEBRWKV) {
#ifdef ENABLE_WEBRWKV
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new web_rwkv_backend,
            [](execution_provider *p) {
                delete (web_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_NCNN) {
#ifdef ENABLE_NCNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new ncnn_rwkv_backend,
            [](execution_provider *p) {
                delete (ncnn_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_LLAMACPP) {
#ifdef ENABLE_LLAMACPP
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new llama_cpp_backend,
            [](execution_provider *p) {
                delete (llama_cpp_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_QNN) {
#ifdef ENABLE_QNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new qnn_backend,
            [](execution_provider *p) {
                delete (qnn_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_MNN) {
#ifdef ENABLE_MNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mnn_rwkv_backend,
            [](execution_provider *p) {
                delete (mnn_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_COREML) {
#ifdef ENABLE_COREML
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new coreml_rwkv_backend,
            [](execution_provider *p) {
                delete (coreml_rwkv_backend*)p;
            });
#endif
    } else {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
    }
    return _backend->init(extra);
}

int runtime::load_model(std::string model_path) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    int ret =  _backend->load_model(model_path);
    if (!ret) {
        LOGI("Loaded model from: %s\n", model_path.c_str());
        LOGI("Model num_layers: %d, num_heads: %d, hidden_size: %d, vocab_size: %d\n",
             _backend->n_layers, _backend->num_heads, _backend->hidden_size, _backend->vocab_size);
    } else {
        LOGE("Failed to load model from: %s, errno = %d\n", model_path.c_str(), ret);
    }

    // Initialize state
    _state_head = new state_node;
    if (_state_head == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    _backend->clear_state();
    _backend->get_state(_state_head->state);

    _vocab_size = _backend->get_num_vocab();
    return ret;
}

int runtime::load_tokenizer(std::string vocab_file) {
    if (_tokenizer != nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
        [](tokenizer_base *p) {
            delete (trie_tokenizer*)p;
        });
    if (_tokenizer == nullptr) {
        return RWKV_ERROR_TOKENIZER;
    }
    return _tokenizer->load(vocab_file);
}

int runtime::load_vision_encoder(std::string model_path, std::string adapter_path) {
#ifdef ENABLE_VISION
    auto adapter_path_cstr = adapter_path.empty() ? NULL : adapter_path.c_str();
    _vision_encoder = std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>>(clip_model_load(model_path.c_str(), adapter_path_cstr, 0),
        [](clip_ctx *p) {
            clip_free(p);
        });
    if (_vision_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::load_whisper_encoder(std::string model_path) {
#ifdef ENABLE_WHISPER
    whisper_context_params cparams = whisper_context_default_params();
    _whisper_encoder = std::unique_ptr<whisper_context, std::function<void(whisper_context*)>>(whisper_init_from_file_with_params(model_path.c_str(), cparams),
        [](whisper_context *p) {
            whisper_free(p);
        });
    if (_whisper_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    whisper_init_state(_whisper_encoder.get());
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::release_vision_encoder() {
#ifdef ENABLE_VISION
    _vision_encoder = nullptr;
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::release_whisper_encoder() {
#ifdef ENABLE_WHISPER
    _whisper_encoder = nullptr;
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

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

int runtime::load_initial_state(std::string state_path) {
    if (state_path.find(".rmpack") == std::string::npos) {
        LOGE("the specified state file is not a rmpack file\n");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    RMPack state_pack(state_path);
    int hidden_size_config = state_pack.getConfig()["hidden_size"];
    auto files = state_pack.getFiles();
    if (files.size() != _backend->n_layers) {
        LOGE("state file has %d layers, but model has %d layers\n", files.size(), _backend->n_layers);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (hidden_size_config != _backend->hidden_size) {
        LOGE("state file has hidden size %d, but model has hidden size %d\n", hidden_size_config, _backend->hidden_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    size_t state_size = state_pack.getFileSize(files[0].filename);

    std::vector<std::vector<half_float::half>> states(_backend->n_layers);
    for (int i = 0; i < _backend->n_layers; i++) {
        states[i].resize(state_size / sizeof(half_float::half));
        auto data = state_pack.readFileToMemory(files[i].filename);
        memcpy(states[i].data(), data, state_size);
        state_pack.freeFileMemory(files[i].filename);
    }
    _backend->load_raw_states(states);

    _backend->get_state(_state_head->state);
    delete_state_node_after(_state_head);
    return RWKV_SUCCESS;
}

void runtime::clear_initial_state() {
    _backend->clear_state();
    _backend->get_state(_state_head->state);
    delete_state_node_after(_state_head);
}

int runtime::eval_logits(int id, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = _backend->eval(id, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits(std::vector<int> ids, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    int ret;
    for (; i + _prefill_chunk_size <= ids.size(); i += _prefill_chunk_size) {
        auto ids_chunk = std::vector<int>(ids.begin() + i, ids.begin() + i + _prefill_chunk_size);
        ret = _backend->eval(ids_chunk, logits);
        if (ret != RWKV_SUCCESS) return ret;
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += _prefill_chunk_size;
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    if (i < ids.size()) {
        auto ids_left = std::vector<int>(ids.begin() + i, ids.end());
        ret = _backend->eval(ids_left, logits);
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

int runtime::eval_logits_with_embeddings(const float *embeddings, int n_tokens, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = _backend->eval_with_embeddings(embeddings, n_tokens, logits);
    auto end = std::chrono::high_resolution_clock::now();
    if (n_tokens > 1) {
        _prefill_speed = n_tokens * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    } else {
        _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return ret;
}

int runtime::chat(std::string input, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    set_is_generating(true);
    _stop_signal = false;
    std::string prompt = input + _eos_token + _response_role + ":";
    if (!_user_role.empty()) {
        prompt = _bos_token + _user_role + ": " + prompt;
    }
    std::vector<int> ids = _tokenizer->encode(prompt);
    float *logits = nullptr;

    _response_buffer = "";
    _response_buffer_ids.clear();
    _response_buffer_eos_found = false;
    _prefill_progress_start(ids.size());
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }
    _prefill_progress_finish();

    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size);

        int idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == 0) {
            break;
        }
        _occurences[idx]++;

        std::string next = _tokenizer->decode(idx);
        _response_buffer += next;
        _response_buffer_ids.push_back(idx);
        if (callback) {
            callback(_response_buffer.c_str(), idx, next.c_str());
        }

        for (auto &stop_code : _stop_codes) {
            if (_response_buffer.size() >= stop_code.size() &&
                _response_buffer.compare(_response_buffer.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                _response_buffer_eos_found = true;
                break;
            }
        }

        ret = eval_logits(idx, logits);
        if (ret) return ret;
        if (_response_buffer_eos_found) break;
        if (_stop_signal) break;
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

std::string runtime::apply_chat_template(std::vector<std::string> inputs, bool enable_reasoning) {
    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    std::string text = _prompt;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 2 == 0) {
            auto user_text = inputs[i];
            user_text = replace_text(user_text, "\r\n", "\n");
            user_text = replace_text(user_text, "\n\n", "\n");

            if (!_user_role.empty()) {
                text += _bos_token + _user_role + ": " + inputs[i] + _eos_token;
            } else {
                text += inputs[i] + _eos_token;
            }
        } else {
            if (i == inputs.size() - 1) {
                text += _response_role + ": " + inputs[i];
            } else {
                text += _response_role + ": " + inputs[i] + _eos_token;
            }
        }
    }

    if (inputs.size() % 2 != 0) {
        text +=  _response_role + ":";
        if (enable_reasoning) {
            text += " " + _thinking_token;
        }
    }
    return text;
}

runtime::state_node* runtime::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = _state_head;
    size_t compare_pos = 0;

    // find the last node that matches the input text
    while (node->next) {

        if (compare_pos + node->next->ids.size() > ids.size() || !std::equal(ids.begin() + compare_pos, ids.begin() + compare_pos + node->next->ids.size(), node->next->ids.begin())) {
            // the text will diverge at next node
            break;
        }
        std::string debug_msg = "matched tokens:";
        for (auto id : node->next->ids) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGI("%s\n", debug_msg.c_str());
        compare_pos += node->next->ids.size();
        node = node->next;
    }

    _backend->set_state(node->state);
    delete_state_node_after(node);

    new_ids_to_prefill = std::vector<int>(ids.begin() + compare_pos, ids.end());
    std::string debug_msg = "new tokens to prefill: ";
    for (auto id : new_ids_to_prefill) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());
    return node;
}

int runtime::register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits) {
    node->next = new state_node;
    if (node->next == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    node = node->next;
    node->ids = std::vector<int>(ids);
    _backend->get_state(node->state);
    node->last_logits.resize(_vocab_size);
    memcpy(node->last_logits.data(), logits, _vocab_size * sizeof(float));
    return RWKV_SUCCESS;
}

int runtime::chat(std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    set_is_generating(true);
    _stop_signal = false;
    _response_buffer.clear();
    _response_buffer_ids.clear();
    _response_buffer_eos_found = false;

    if (_prefilling_thread.joinable() && _prefilling_thread.get_id() != std::this_thread::get_id()) {
        LOGD("Found prefilling thread, joining\n");
        _prefilling_thread.join();
        LOGD("_prefilling_thread finished.\n");
    }

    auto input_text = apply_chat_template(inputs, enable_reasoning);
    LOGD("Applied chat template: \"%s\"\n", input_text.c_str());
    std::vector<int> text_ids = _tokenizer->encode(input_text);
    std::string debug_msg = "text_ids: ";
    for (auto id : text_ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    float *logits = nullptr;
    std::vector<int> tokens_to_prefill;
    auto node = match_and_load_state(text_ids, tokens_to_prefill);

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
            eval_logits(tokens_to_prefill_chunk, logits);
            int ret = register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
            if (ret) return ret;
            _backend->free_logits_if_allocated(logits);
        }
    }
    _prefill_progress_finish();

    _response_buffer = input_text.substr(input_text.rfind(_response_role + ":") + (_response_role + ":").size());
    std::vector<int> response_ids_raw;
    _response_buffer_ids = _tokenizer->encode(_response_buffer);
    int ret;

    _occurences.clear();
    for (int i = 1; i < inputs.size(); i += 2) {
        std::vector<int> ids = _tokenizer->encode(" " + inputs[i]);
        for (auto id: ids) {
            for (auto &[_id, occurence] : _occurences) {
                _occurences[_id] *= _penalty_decay;
            }
            _occurences[id]++;
        }
    }

    if (logits == nullptr) {
        if (node->last_logits.size() == _vocab_size) {
            logits = node->last_logits.data();
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            ret = eval_logits(text_ids.back(), logits);
            if (ret) return ret;
            response_ids_raw.emplace_back(text_ids.back());
        }
    }

    int decoded_idx = 0;
    bool thinking_end_tag_found = false;
    bool is_pseudo_thinking = enable_reasoning && _response_buffer.find("</think>") != std::string::npos;
    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size);

        if (is_pseudo_thinking && i == 0) {
            // token 61 is '<', 261 is '\n\n'
            logits[61] = -1e9f;
            logits[261] = -1e9f;
        } else if (is_pseudo_thinking && i == 1 && decoded_idx == 11) {
            logits[61] = -1e9f;
        }

        decoded_idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        if (decoded_idx == 0) {
            break;
        }

        std::string decoded = _tokenizer->decode(decoded_idx);
        std::string tmp = _response_buffer + decoded;
        for (auto &stop_code : _stop_codes) {
            if (enable_reasoning && !thinking_end_tag_found && stop_code == "\n\n") {
                continue;
            }
            if (tmp.size() >= stop_code.size() &&
                tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                LOGD("stop code found: %s\n", stop_code.c_str());
                _response_buffer_eos_found = true;
                break;
            }
        }

        if (enable_reasoning && !thinking_end_tag_found) {
            if (tmp.find("</think>") != std::string::npos) {
                thinking_end_tag_found = true;
            }
        }

        if (_response_buffer_eos_found || _stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", _response_buffer_eos_found, _stop_signal);
            break;
        }

        if (i != 0 || logits != node->last_logits.data()) {
            _backend->free_logits_if_allocated(logits);
        }
        ret = eval_logits(decoded_idx, logits);
        if (ret) return ret;

        response_ids_raw.emplace_back(decoded_idx);
        _response_buffer += decoded;
        _response_buffer_ids.emplace_back(decoded_idx);
        if (i == 0 && _response_buffer[0] == ' ') {
            _response_buffer = _response_buffer.substr(1);
        }

        _occurences[decoded_idx]++;
        if (callback) {
            callback(_response_buffer.c_str(), decoded_idx, decoded.c_str());
        }
    }

    if (response_ids_raw.size() > 0) {
        int ret = register_state_checkpoint(node, response_ids_raw, logits);
        if (ret) return ret;
    }

    if (logits != node->last_logits.data()) {
        _backend->free_logits_if_allocated(logits);
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::set_prompt(std::string prompt) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    LOGD("Setting and processing prompt: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = _tokenizer->encode(prompt);
    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->set_state(_state_head->state);
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
    }
    float *logits = nullptr;
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}

std::string runtime::get_prompt() {
    return _prompt;
}

#ifdef ENABLE_VISION
int runtime::set_image_prompt(std::string path) {
    if (_backend == nullptr || _tokenizer == nullptr || _vision_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<img src=\"" + path + "\">";
    std::vector<int> ids = _tokenizer->encode(prompt);

    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->set_state(_state_head->state);
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto embd = llava_image_embed_make_with_filename(_vision_encoder.get(), 4, path.c_str());
    if (embd == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("siglip duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    float *logits = nullptr;

    start = std::chrono::high_resolution_clock::now();
    int ret = eval_logits_with_embeddings(embd->embed, embd->n_image_pos, logits);
    if (ret) {
        return ret;
    }
    end = std::chrono::high_resolution_clock::now();
    LOGI("eval_logits_with_embeddings duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
    llava_image_embed_free(embd);
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_WHISPER
int runtime::set_audio_prompt(std::string path) {
    if (_backend == nullptr || _tokenizer == nullptr || _whisper_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<audio src=\"" + path + "\">";
    std::vector<int> ids = _tokenizer->encode(prompt);

    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->set_state(_state_head->state);
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
    }

    wav_file wav;
    wav.load(path);

    auto start = std::chrono::high_resolution_clock::now();
    whisper_pcm_to_mel(_whisper_encoder.get(), wav.samples.data(), wav.samples.size(), 4);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("whisper_pcm_to_mel time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::high_resolution_clock::now();
    whisper_encode(_whisper_encoder.get(), 0, 4);
    end = std::chrono::high_resolution_clock::now();
    LOGI("whisper_encode time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    float *logits = nullptr;

    auto embd = whisper_get_adapter_output_tensor(_whisper_encoder.get());

    int ret = eval_logits_with_embeddings((const float *)embd->data, embd->ne[1], logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
static void save_samples_to_wav(std::vector<float> samples, std::string path, int sample_rate = 24000) {
    wav_file wav_file;
    wav_file.sample_rate = sample_rate;
    wav_file.num_channels = 1;
    wav_file.num_samples = samples.size();
    wav_file.bit_depth = 16;
    wav_file.audio_format = 1;
    wav_file.byte_rate = sample_rate * 16 / 8;
    wav_file.block_align = 2;
    wav_file.samples = samples;
    wav_file.save(path);
}

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

int runtime::run_spark_tts(std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

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
    auto text_tokens = tokenizer_encode(full_text);
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

    clear_state();
    float *logits = nullptr;
    int ret = eval_logits(input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (int i = 0; i < tts_max_length; i++) {
        int idx = _sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == tts_eos_token) {
            break;
        }

        output_tokens.push_back(idx);
        ret = eval_logits(idx, logits);
        if (ret || !logits) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] LLM inference time: %lf ms", duration);
    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed());
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed());

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_streaming(std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(text);
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
            auto text_tokens = tokenizer_encode(full_text);
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

            clear_state();
            float *logits = nullptr;
            int ret = eval_logits(input_tokens, logits);
            if (ret || !logits) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }
            logits[tts_eos_token] = -1e9;

            for (int i = 0; i < tts_max_length; i++) {
                int idx = _sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                _backend->free_logits_if_allocated(logits);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                ret = eval_logits(idx, logits);
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

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed());
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed());

    llm_inference_thread.join();
    detokenize_thread.join();
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration);
    LOGI("[TTS] Output audio length: %lf s", _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] RTF (%s): %lf", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] TTFA (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", ttfa);
    LOGI("\n\n");

    set_is_generating(false);
    return RWKV_SUCCESS;
}
#endif

int runtime::gen_completion(std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *)) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _response_buffer = "";
    _response_buffer_ids.clear();
    _response_buffer_eos_found = false;
    set_is_generating(true);
    _stop_signal = false;

    std::vector<int> ids = _tokenizer->encode(prompt);
    _prefill_progress_start(ids.size());

    float *logits = nullptr;
    int ret = eval_logits(ids, logits);
    if (ret || !logits) {
        set_is_generating(false);
        return ret;
    }
    _prefill_progress_finish();

    _response_buffer = prompt;
    _response_buffer_ids = ids;
    static int idx = 0;
    bool apply_penalties = _presence_penalty > 0.0f && _frequency_penalty > 0.0f && _penalty_decay > 0.0f;
    for (int i = 0; i < max_length; i++) {
        if (apply_penalties) {
            apply_logits_penalties(logits, _vocab_size);
        }

        idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        _response_buffer_eos_found = (idx == stop_code);

        std::string next = _tokenizer->decode(idx);
        _response_buffer += next;
        _response_buffer_ids.push_back(idx);
        ret = eval_logits(idx, logits);
        if (callback) {
            callback(_response_buffer.c_str(), idx, next.c_str());
        }

        if (_response_buffer_eos_found || _stop_signal) {
            break;
        }

        if (apply_penalties) {
            _occurences[idx]++;
        }
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

double runtime::get_avg_decode_speed() {
    double speed_from_backend = _backend->get_decode_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_decode_speed < 0) {
        return 0.0;
    } else {
        return _decode_speed;
    }
}

double runtime::get_avg_prefill_speed() {
    double speed_from_backend = _backend->get_prefill_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_prefill_speed < 0) {
        return 0.0;
    } else {
        return _prefill_speed;
    }
}

} // namespace rwkvmobile

