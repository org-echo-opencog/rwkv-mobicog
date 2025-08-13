#ifndef BACKEND_H
#define BACKEND_H

#include <string>
#include <vector>
#include <any>
#include "half.hpp"

#include "commondef.h"

namespace rwkvmobile {

class execution_provider {
public:
    virtual int init(void * extra) { return 0; }
    virtual int init(std::string model_path, void * extra) { return 0; }
    virtual int load_model(std::string model_path) { return RWKV_ERROR_MODEL; }
    virtual int eval(int id, float *& logits) { return 0; };
    virtual int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) { return 0; };
    virtual int eval_with_embeddings(const float *embeddings, int n_tokens, float *& logits) { return RWKV_ERROR_UNSUPPORTED; };
    virtual void free_logits_if_allocated(float *& logits) { return; };
    virtual int get_state(std::any &state) { return 0; }
    virtual int set_state(std::any state) { return 0; }
    virtual int free_state(std::any state) { return 0; }
    virtual int clear_state() { return 0; }
    virtual int release_model() { return 0; };
    virtual int release() { return 0; };
    virtual bool is_available() { return false; };

    virtual double get_prefill_speed() { return -1; }
    virtual double get_decode_speed() { return -1; }

    virtual int load_raw_states(std::vector<std::vector<half_float::half>> states) { return RWKV_ERROR_UNSUPPORTED; };

    int get_head_count() { return num_heads; }
    int get_hidden_size() { return hidden_size; }
    int get_num_vocab() { return vocab_size; }
    int get_version() { return version; }

    int n_layers;
    int num_heads;
    int hidden_size;
    int version;
    int vocab_size;

    std::string extra_str;
};

enum {
    RWKV_BACKEND_WEBRWKV = 0,
    RWKV_BACKEND_NCNN,
    RWKV_BACKEND_LLAMACPP,
    RWKV_BACKEND_QNN,
    RWKV_BACKEND_MNN,
    RWKV_BACKEND_COREML,
    RWKV_BACKEND_COUNT,
};

std::string backend_enum_to_str(int backend);
int backend_str_to_enum(std::string backend);

}

#endif