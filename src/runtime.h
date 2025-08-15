#ifndef RUNTIME_H
#define RUNTIME_H

#include <string>
#include <map>
#include <memory>
#include <functional>
#include <cstdlib>
#include <any>
#include <thread>
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"
#include "soc_detect.h"

#include "logger.h"
#include "embedding/rwkv_embedding.h"

#ifdef ENABLE_VISION
#include "clip.h"
#endif

#ifdef ENABLE_WHISPER
#include "whisper.h"
#endif

#ifdef ENABLE_TTS
#include "sparktts.h"
#if !defined(_WIN32)
#include "kaldifst/csrc/text-normalizer.h"
#endif
#endif

namespace rwkvmobile {

class runtime {
public:
    runtime() {
#ifdef __ANDROID__
        setenv("KMP_DUPLICATE_LIB_OK", "1", 1);
#endif
        _soc_detect.detect_platform();
    };
    ~runtime() {};
    int init(std::string backend_name);
    int init(int backend_id);
    int init(std::string backend_name, void * extra);
    int init(int backend_id, void * extra);
    int load_model(std::string model_path);
    int load_tokenizer(std::string vocab_file);
    int load_vision_encoder(std::string model_path, std::string adapter_path = "");
    int load_whisper_encoder(std::string model_path);
    int eval_logits(int id, float *& logits);
    int eval_logits(std::vector<int> ids, float *& logits);
    int eval_logits_with_embeddings(const float *embeddings, int n_tokens, float *& logits);
    void free_logits_if_allocated(float *& logits) {
        if (_backend != nullptr) {
            _backend->free_logits_if_allocated(logits);
        }
    }

    // without history
    int chat(std::string input, const int max_length, void (*callback)(const char *, const int, const char *) = nullptr, bool enable_reasoning = false);

    // with history
    int chat(std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int, const char *) = nullptr, bool enable_reasoning = false);
    int gen_completion(std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *));

    int set_prompt(std::string prompt);
    std::string get_prompt();

    int load_initial_state(std::string state_path);
    void clear_initial_state();

    std::string get_response_buffer_content() { return _response_buffer; }
    const std::vector<int32_t> get_response_buffer_ids() { return _response_buffer_ids; }
    void clear_response_buffer() { _response_buffer = ""; _response_buffer_ids.clear(); }
    bool get_response_buffer_eos_found() { return _response_buffer_eos_found; }
#ifdef ENABLE_VISION
    int set_image_prompt(std::string path);
#endif

#ifdef ENABLE_WHISPER
    int set_audio_prompt(std::string path);
#endif

#ifdef ENABLE_TTS
    int sparktts_load_models(
        std::string wav2vec2_path,
        std::string bicodec_tokenizer_path,
        std::string bicodec_detokenizer_path
    );

    int sparktts_release_models();

    int run_spark_tts(std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path);
    int run_spark_tts_streaming(std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path);

    std::vector<float>& tts_get_streaming_buffer() {
        return _tts_output_samples_buffer;
    }

    int tts_register_text_normalizer(std::string path) {
#if !defined(_WIN32)
        if (!std::ifstream(path).good()) {
            LOGE("[TTS] Failed to load text normalizer file %s\n", path.c_str());
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _tn_list.push_back(std::make_unique<kaldifst::TextNormalizer>(path));
        LOGI("[TTS] Loaded text normalizer file %s\n", path.c_str());
#endif
        return RWKV_SUCCESS;
    }

    int tts_clear_text_normalizer() {
#if !defined(_WIN32)
        _tn_list.clear();
#endif
        return RWKV_SUCCESS;
    }
#endif

    // for state management
    int clear_state() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }

        if (sampler != nullptr) {
            sampler->clear_occurences();
        }
        _backend->clear_state();
        return RWKV_SUCCESS;
    }

    int release() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        clear_state();
        int ret = _backend->release_model();
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
        _tokenizer = nullptr;
        sampler = nullptr;
        return _backend->release();
    }

    int release_vision_encoder();
    int release_whisper_encoder();

    inline int set_seed(int64_t seed) {
        if (sampler == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        sampler->set_seed(seed);
        _seed = seed;
        return 0;
    }

    inline int64_t get_seed() { return _seed; }

    inline void set_user_role(std::string role) { _user_role = role; }
    inline void set_response_role(std::string role) { _response_role = role; }
    inline void set_bos_token(std::string token) { _bos_token = token; }
    inline void set_eos_token(std::string token) {
        _eos_token = token;
        _stop_codes[0] = _eos_token;
    }
    std::string get_user_role() { return _user_role; }
    std::string get_response_role() { return _response_role; }
    std::string get_bos_token() { return _bos_token; }
    std::string get_eos_token() { return _eos_token; }

    std::string apply_chat_template(std::vector<std::string> inputs, bool enable_reasoning = false);

    int get_vocab_size() { return _vocab_size; }

    inline std::vector<std::string> get_stop_codes() { return _stop_codes; }
    inline void set_stop_codes(std::vector<std::string> stop_codes) { _stop_codes = stop_codes; }
    inline std::string get_thinking_token() { return _thinking_token; }
    inline void set_thinking_token(std::string thinking_token) { _thinking_token = thinking_token; }

    inline void set_sampler_params(float temperature, int top_k, float top_p) {
        if (sampler == nullptr) {
            LOGE("Sampler not initialized\n");
            return;
        }
        LOGD("Setting sampler params: temperature=%f, top_k=%d, top_p=%f\n", temperature, top_k, top_p);
        sampler->set_temperature(temperature);
        sampler->set_top_k(top_k);
        sampler->set_top_p(top_p);
    }

    inline void set_penalty_params(float presence_penalty, float frequency_penalty, float penalty_decay) {
        if (sampler == nullptr) {
            LOGE("Sampler not initialized\n");
            return;
        }
        LOGD("Setting penalty params: presence_penalty=%f, frequency_penalty=%f, penalty_decay=%f\n", presence_penalty, frequency_penalty, penalty_decay);
        sampler->set_presence_penalty(presence_penalty);
        sampler->set_frequency_penalty(frequency_penalty);
        sampler->set_penalty_decay(penalty_decay);
    }

    void set_token_banned(std::vector<int> token_banned) {
        if (sampler == nullptr) {
            LOGE("Sampler not initialized\n");
            return;
        }
        sampler->set_token_banned(token_banned);
    }

    inline bool is_generating() { return _is_generating; }
    inline void set_is_generating(bool is_generating) { _is_generating = is_generating; }

    inline bool get_stop_signal() { return _stop_signal; }
    inline void set_stop_signal(bool stop_signal) { _stop_signal = stop_signal; }

    std::string get_available_backends_str();
    int get_available_backend_ids(std::vector<int> &backend_ids);

    double get_avg_decode_speed();
    double get_avg_prefill_speed();
    double get_prefill_progress() { return _prefill_progress; }

    int load_embedding_model(std::string model_path) {
        if (_embedding == nullptr) {
            _embedding = std::make_unique<rwkv_embedding>();
        }
        return _embedding->load_model(model_path);
    }

    int release_embedding_model() {
        if (_embedding == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _embedding->release();
        _embedding = nullptr;
        return RWKV_SUCCESS;
    }

    int load_rerank_model(std::string model_path) {
        if (_embedding == nullptr) {
            _embedding = std::make_unique<rwkv_embedding>();
        }
        return _embedding->load_rerank_model(model_path);
    }

    std::vector<std::vector<float>> get_embedding(std::vector<std::string> inputs)  {
        if (_embedding == nullptr) {
            LOGE("Embedding model not loaded\n");
            return {};
        }
        return _embedding->get_embeddings(inputs);
    }

    std::vector<float> rerank(std::string query, const std::vector<std::string> &chunks)  {
        if (_embedding == nullptr) {
            LOGE("Embedding model not loaded\n");
            return {};
        }
        return _embedding->rerank(query, chunks);
    }

    // platform info
    const char * get_platform_name() {
        auto platform_name = _soc_detect.get_platform_name();
        LOGD("Platform name: %s", platform_name);
        return platform_name;
    }

    const char * get_soc_name() {
        auto soc_name = _soc_detect.get_soc_name();
        LOGD("SOC name: %s", soc_name);
        return soc_name;
    }

    const char * get_soc_partname() {
        auto soc_partname = _soc_detect.get_soc_partname();
        LOGD("SOC partname: %s", soc_partname);
        return soc_partname;
    }

    // backend
    std::string backend_id_to_str(int backend_id) {
        return backend_enum_to_str(backend_id);
    }
    int backend_str_to_id(std::string backend_str) {
        return backend_str_to_enum(backend_str);
    }

    void backend_set_extra_str(std::string str) {
        _backend->extra_str = str;
    }

    // tokenizer
    std::vector<int> tokenizer_encode(std::string text) {
        if (_tokenizer == nullptr) {
            return {};
        }
        return _tokenizer->encode(text);
    }

    std::string tokenizer_decode(std::vector<int> ids) {
        if (_tokenizer == nullptr) {
            return "";
        }
        return _tokenizer->decode(ids);
    }

    std::string tokenizer_decode(int id) {
        if (_tokenizer == nullptr) {
            return "";
        }
        return _tokenizer->decode(id);
    }

    // sampler
    int sampler_sample(std::vector<float> logits) {
        if (sampler == nullptr) {
            return -1;
        }
        return sampler->sample(logits.data(), logits.size());
    }

    inline void set_cache_dir(std::string cache_dir) { _cache_dir = cache_dir; }

    std::unique_ptr<NucleusSampler> sampler;

private:
    std::unique_ptr<execution_provider, std::function<void(execution_provider*)>> _backend;
    std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>> _tokenizer;
    std::unique_ptr<rwkv_embedding> _embedding;

    double _prefill_speed = -1;
    double _decode_speed = -1;

    const int _prefill_chunk_size = 64;

    int _current_prefill_total_tokens = -1;
    int _current_prefill_finished_tokens = 0;
    double _prefill_progress = 0.0;

    void _prefill_progress_start(int total_tokens) {
        _current_prefill_total_tokens = total_tokens;
        _current_prefill_finished_tokens = 0;
        _prefill_progress = 0;
    }

    void _prefill_progress_finish() {
        _current_prefill_total_tokens = -1;
        _prefill_progress = 1.0;
    }

    std::string _cache_dir = "";

    soc_detect _soc_detect;

    int _vocab_size = 0;

    int64_t _seed = 42;
    std::string _user_role = "User";
    std::string _response_role = "Assistant";
    std::string _prompt;
    std::string _thinking_token = "<think";

    bool _is_generating = false;
    bool _stop_signal = false;

    std::thread _prefilling_thread;

    std::vector<std::string> _stop_codes = {"\n\n", "\nUser", "User"};
    std::string _bos_token = "";
    std::string _eos_token = "\n\n";

    std::string _response_buffer;
    std::vector<int32_t> _response_buffer_ids;
    bool _response_buffer_eos_found = false;

#ifdef ENABLE_VISION
    std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>> _vision_encoder;
#endif

#ifdef ENABLE_WHISPER
    std::unique_ptr<whisper_context, std::function<void(whisper_context*)>> _whisper_encoder;
#endif

#ifdef ENABLE_TTS
    std::unique_ptr<sparktts> _sparktts;
#if !defined(_WIN32)
    std::vector<std::unique_ptr<kaldifst::TextNormalizer>> _tn_list;
#endif

    std::vector<float> _tts_output_samples_buffer;
#endif
};

}

#endif
