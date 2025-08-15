#include "runtime.h"
#include "commondef.h"
#include "c_api.h"
#include "logger.h"
#include "soc_detect.h"
#include <cstring>
#include <cstdlib>
#include <thread>

namespace rwkvmobile {

extern "C" {

rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name) {
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return nullptr;
    }
    rt->init(backend_name);
    return rt;
}

rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name_extra(const char * backend_name, void * extra) {
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return nullptr;
    }
    rt->init(backend_name, extra);
    return rt;
}

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t handle, const char * model_path) {
    if (handle == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_model(model_path);
}

int rwkvmobile_runtime_release(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    int ret = rt->release();
    delete rt;
    return ret;
}

int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t handle, const char * vocab_file) {
    if (handle == nullptr || vocab_file == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_tokenizer(vocab_file);
}

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t handle, const int * ids, int ids_len, float * logits, int logits_len) {
    if (handle == nullptr || ids == nullptr || logits == nullptr || ids_len <= 0 || logits_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    std::vector<int> ids_vec(ids, ids + ids_len);
    float *logits_ret = nullptr;
    auto ret = rt->eval_logits(ids_vec, logits_ret);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    memcpy(logits, logits_ret, logits_len * sizeof(float));
    rt->free_logits_if_allocated(logits_ret);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_eval_chat_async(
    rwkvmobile_runtime_t handle,
    const char * input,
    const int max_tokens,
    void (*callback)(const char *, const int, const char *),
    int enable_reasoning) {
    if (handle == nullptr || input == nullptr || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    rt->set_is_generating(true);
    std::thread generation_thread([=]() {
        int ret = rt->chat(
            std::string(input),
            max_tokens,
            callback,
            enable_reasoning != 0);
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_eval_chat_with_history_async(
    rwkvmobile_runtime_t handle,
    const char ** inputs,
    const int num_inputs,
    const int max_tokens,
    void (*callback)(const char *, const int, const char *),
    int enable_reasoning) {
    if (handle == nullptr || inputs == nullptr || num_inputs == 0 || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    rt->set_is_generating(true);
    rt->set_stop_signal(false);
    std::vector<std::string> inputs_vec;
    for (int i = 0; i < num_inputs; i++) {
        inputs_vec.push_back(std::string(inputs[i]));
    }

    std::thread generation_thread([=]() {
        int ret = rt->chat(
            inputs_vec,
            max_tokens,
            callback,
            enable_reasoning != 0);
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_gen_completion_async(
    rwkvmobile_runtime_t handle,
    const char * prompt,
    const int max_tokens,
    const int stop_code,
    void (*callback)(const char *, const int, const char *)) {
    if (handle == nullptr || prompt == nullptr || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    rt->clear_response_buffer();
    rt->set_is_generating(true);
    rt->set_stop_signal(false);
    std::thread generation_thread([=]() {
        int ret = rt->gen_completion(
            std::string(prompt),
            max_tokens,
            stop_code,
            callback);
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_gen_completion(
    rwkvmobile_runtime_t handle,
    const char * prompt,
    const int max_tokens,
    const int stop_code,
    void (*callback)(const char *, const int, const char *)) {
    if (handle == nullptr || prompt == nullptr || max_tokens <= 0 || callback == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    rt->clear_response_buffer();
    return rt->gen_completion(
        std::string(prompt),
        max_tokens,
        stop_code,
        callback);
}

int rwkvmobile_runtime_stop_generation(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_stop_signal(true);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_is_generating(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    bool is_generating = rt->is_generating();
    return is_generating;
}

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->clear_state();
}

int rwkvmobile_runtime_load_initial_state(rwkvmobile_runtime_t handle, const char * state_path) {
    if (handle == nullptr || state_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_initial_state(state_path);
}

void rwkvmobile_runtime_clear_initial_state(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(handle);
    rt->clear_initial_state();
}

int rwkvmobile_runtime_get_available_backend_names(char * backend_names_buffer, int buffer_size) {
    if (backend_names_buffer == nullptr || buffer_size <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    auto backend_names = rt->get_available_backends_str();
    if (backend_names.size() >= buffer_size) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(backend_names_buffer, backend_names.c_str(), buffer_size);
    delete rt;
    return RWKV_SUCCESS;
}

struct sampler_params rwkvmobile_runtime_get_sampler_params(rwkvmobile_runtime_t runtime) {
    struct sampler_params params;
    params.temperature = 0;
    params.top_k = 0;
    params.top_p = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class runtime *>(runtime);
    params.temperature = rt->sampler->get_temperature();
    params.top_k = rt->sampler->get_top_k();
    params.top_p = rt->sampler->get_top_p();
    return params;
}

void rwkvmobile_runtime_set_sampler_params(rwkvmobile_runtime_t runtime, struct sampler_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_sampler_params(params.temperature, params.top_k, params.top_p);
}

struct penalty_params rwkvmobile_runtime_get_penalty_params(rwkvmobile_runtime_t runtime) {
    struct penalty_params params;
    params.presence_penalty = 0;
    params.frequency_penalty = 0;
    params.penalty_decay = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class runtime *>(runtime);
    params.presence_penalty = rt->sampler->get_presence_penalty();
    params.frequency_penalty = rt->sampler->get_frequency_penalty();
    params.penalty_decay = rt->sampler->get_penalty_decay();
    return params;
}

void rwkvmobile_runtime_set_penalty_params(rwkvmobile_runtime_t runtime, struct penalty_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_penalty_params(params.presence_penalty, params.frequency_penalty, params.penalty_decay);
}

int rwkvmobile_runtime_set_prompt(rwkvmobile_runtime_t runtime, const char * prompt) {
    if (runtime == nullptr || prompt == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->set_prompt(prompt);
}

int rwkvmobile_runtime_get_prompt(rwkvmobile_runtime_t runtime, char * prompt, const int buf_len) {
    if (runtime == nullptr || prompt == nullptr || buf_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    std::string prompt_str = rt->get_prompt();
    if (prompt_str.size() >= buf_len) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(prompt, prompt_str.c_str(), buf_len);
    return RWKV_SUCCESS;
}

void rwkvmobile_runtime_add_adsp_library_path(const char * path) {
#ifndef _WIN32
    auto ld_lib_path_char = getenv("LD_LIBRARY_PATH");
    std::string ld_lib_path;
    if (ld_lib_path_char) {
        ld_lib_path = std::string(path) + ":" + std::string(ld_lib_path_char);
    } else {
        ld_lib_path = std::string(path);
    }
    LOGI("Setting LD_LIBRARY_PATH to %s\n", ld_lib_path.c_str());
    setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", path, 1);
#endif
}

void rwkvmobile_runtime_set_qnn_library_path(rwkvmobile_runtime_t runtime, const char * path) {
#ifndef _WIN32
    auto ld_lib_path_char = getenv("LD_LIBRARY_PATH");
    std::string ld_lib_path;
    if (ld_lib_path_char) {
        ld_lib_path = std::string(path) + ":" + std::string(ld_lib_path_char);
    } else {
        ld_lib_path = std::string(path);
    }
    LOGI("Setting LD_LIBRARY_PATH to %s\n", ld_lib_path.c_str());
    setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", path, 1);
#endif
    if (runtime == nullptr || path == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->backend_set_extra_str(path);
}

double rwkvmobile_runtime_get_avg_decode_speed(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->get_avg_decode_speed();
}

double rwkvmobile_runtime_get_avg_prefill_speed(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->get_avg_prefill_speed();
}

int rwkvmobile_runtime_load_vision_encoder(rwkvmobile_runtime_t runtime, const char * encoder_path) {
#if ENABLE_VISION
    if (runtime == nullptr || encoder_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->load_vision_encoder(encoder_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_load_vision_encoder_and_adapter(rwkvmobile_runtime_t runtime, const char * encoder_path, const char * adapter_path) {
#if ENABLE_VISION
    if (runtime == nullptr || encoder_path == nullptr || adapter_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->load_vision_encoder(encoder_path, adapter_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_release_vision_encoder(rwkvmobile_runtime_t runtime) {
#if ENABLE_VISION
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->release_vision_encoder();
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_image_prompt(rwkvmobile_runtime_t runtime, const char * image_path) {
#if ENABLE_VISION
    if (runtime == nullptr || image_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->set_image_prompt(image_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_load_whisper_encoder(rwkvmobile_runtime_t runtime, const char * encoder_path) {
#if ENABLE_WHISPER
    if (runtime == nullptr || encoder_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->load_whisper_encoder(encoder_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_release_whisper_encoder(rwkvmobile_runtime_t runtime) {
#if ENABLE_WHISPER
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->release_whisper_encoder();
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_audio_prompt(rwkvmobile_runtime_t runtime, const char * audio_path) {
#if ENABLE_WHISPER
    if (runtime == nullptr || audio_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->set_audio_prompt(audio_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_token_banned(rwkvmobile_runtime_t runtime, const int * token_banned, int token_banned_len) {
    if (runtime == nullptr || token_banned == nullptr || token_banned_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    std::vector<int> token_banned_vec(token_banned, token_banned + token_banned_len);
    rt->set_token_banned(token_banned_vec);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_eos_token(rwkvmobile_runtime_t runtime, const char * eos_token) {
    if (runtime == nullptr || eos_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_eos_token(eos_token);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_bos_token(rwkvmobile_runtime_t runtime, const char * bos_token) {
    if (runtime == nullptr || bos_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_bos_token(bos_token);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_user_role(rwkvmobile_runtime_t runtime, const char * user_role) {
    if (runtime == nullptr || user_role == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_user_role(user_role);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_response_role(rwkvmobile_runtime_t runtime, const char * response_role) {
    if (runtime == nullptr || response_role == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_response_role(response_role);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_thinking_token(rwkvmobile_runtime_t runtime, const char * thinking_token) {
    if (runtime == nullptr || thinking_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_thinking_token(thinking_token);
    return RWKV_SUCCESS;
}

struct response_buffer rwkvmobile_runtime_get_response_buffer_content(rwkvmobile_runtime_t runtime) {
    struct response_buffer buffer;
    buffer.content = nullptr;
    buffer.length = 0;
    buffer.eos_found = false;
    if (runtime == nullptr) {
        return buffer;
    }
    auto rt = static_cast<class runtime *>(runtime);
    std::string content = rt->get_response_buffer_content();
    buffer.length = content.size();
    buffer.content = (char *)malloc(buffer.length * sizeof(char));
    if (buffer.content == nullptr) {
        return buffer;
    }
    memset(buffer.content, 0, buffer.length);
    strncpy(buffer.content, content.c_str(), buffer.length);
    buffer.eos_found = rt->get_response_buffer_eos_found();
    return buffer;
}

void rwkvmobile_runtime_free_response_buffer(struct response_buffer buffer) {
    if (buffer.content == nullptr) {
        return;
    }
    free((void *)buffer.content);
}

struct token_ids rwkvmobile_runtime_get_response_buffer_ids(rwkvmobile_runtime_t runtime) {
    struct token_ids ids;
    ids.ids = nullptr;
    ids.len = 0;
    if (runtime == nullptr) {
        return ids;
    }
    auto rt = static_cast<class runtime *>(runtime);
    auto ids_vec = rt->get_response_buffer_ids();
    ids.ids = (int32_t *)malloc(ids_vec.size() * sizeof(int32_t));
    if (ids.ids == nullptr) {
        return ids;
    }
    for (int i = 0; i < ids_vec.size(); i++) {
        ids.ids[i] = ids_vec[i];
    }
    ids.len = ids_vec.size();
    return ids;
}

void rwkvmobile_runtime_free_token_ids(struct token_ids ids) {
    if (ids.ids == nullptr) {
        return;
    }
    free(ids.ids);
}

int rwkvmobile_runtime_sparktts_load_models(rwkvmobile_runtime_t runtime, const char * wav2vec2_path, const char * bicodec_tokenizer_path, const char * bicodec_detokenizer_path) {
#if ENABLE_TTS
    if (runtime == nullptr || wav2vec2_path == nullptr || bicodec_tokenizer_path == nullptr || bicodec_detokenizer_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->sparktts_load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_sparktts_release_models(rwkvmobile_runtime_t runtime) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->sparktts_release_models();
#else
    return RWKV_SUCCESS;
#endif
}

int rwkvmobile_runtime_run_spark_tts_streaming_async(rwkvmobile_runtime_t runtime, const char * tts_text, const char * prompt_audio_text, const char * prompt_audio_path, const char * output_wav_path) {
#if ENABLE_TTS
    if (runtime == nullptr || tts_text == nullptr || prompt_audio_path == nullptr || output_wav_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_is_generating(true);
    rt->set_stop_signal(false);
    std::string prompt_audio_text_str;
    if (prompt_audio_text == nullptr) {
        prompt_audio_text_str = "";
    } else {
        prompt_audio_text_str = std::string(prompt_audio_text);
    }
    std::thread generation_thread([=]() {
        int ret = rt->run_spark_tts_streaming(tts_text, prompt_audio_text_str, prompt_audio_path, output_wav_path);
        return ret;
    });

    generation_thread.detach();
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

struct tts_streaming_buffer rwkvmobile_runtime_get_tts_streaming_buffer(rwkvmobile_runtime_t runtime) {
    struct tts_streaming_buffer ret;
#if ENABLE_TTS
    auto rt = static_cast<class runtime *>(runtime);
    auto buffer = rt->tts_get_streaming_buffer();
    ret.samples = buffer.data();
    ret.length = buffer.size();
#else
    ret.samples = nullptr;
    ret.length = 0;
#endif
    return ret;
}

int rwkvmobile_runtime_tts_register_text_normalizer(rwkvmobile_runtime_t runtime, const char * path) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->tts_register_text_normalizer(std::string(path));
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

float rwkvmobile_runtime_get_prefill_progress(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return 0;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->get_prefill_progress();
}

const char * rwkvmobile_get_platform_name() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_platform_name();
}

const char * rwkvmobile_get_soc_name() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_soc_name();
}

const char * rwkvmobile_get_soc_partname() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_soc_partname();
}

const char * rwkvmobile_get_htp_arch() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_htp_arch();
}

const char * rwkvmobile_dump_log() {
    return logger_get_log().c_str();
}

void rwkvmobile_set_loglevel(int loglevel) {
    logger_set_loglevel(loglevel);
}

void rwkvmobile_set_cache_dir(rwkvmobile_runtime_t runtime, const char * cache_dir) {
    if (cache_dir == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_cache_dir(std::string(cache_dir));
}

    int rwkvmobile_load_embedding_model(rwkvmobile_runtime_t runtime, const char *model_path) {
    if (runtime == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->load_embedding_model(model_path);
}

int rwkvmobile_load_rerank_model(rwkvmobile_runtime_t runtime, const char *model_path) {
    if (runtime == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->load_rerank_model(model_path);
}

int rwkvmobile_get_embedding(rwkvmobile_runtime_t runtime, const char **input, const int input_length, float **embedding) {
    if (runtime == nullptr || input == nullptr || embedding == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    std::vector<std::string> inputs;
    for (int i = 0; i < input_length; i++) {
        inputs.emplace_back(input[i]);
    }

    auto ebd = rt->get_embedding(inputs);
    if (ebd.empty()) {
        return RWKV_ERROR_EVAL;
    }
    memcpy(embedding, ebd.data(), ebd[0].size() * ebd.size() * sizeof(float));
    return 0;
}

} // extern "C"
} // namespace rwkvmobile
