#ifndef BACKEND_H
#define BACKEND_H

#include <string>
#include <vector>
#include <memory>
#include <any>
#include "half.hpp"

#include "commondef.h"
#include "state.h"

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
    virtual int zero_state() { return 0; }
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

    std::unique_ptr<state_node> state_head;

    int clear_state() {
        if (state_head != nullptr) {
            state_head->delete_after();
            set_state(state_head->state);
        } else {
            zero_state();
            state_head = std::make_unique<state_node>();
            get_state(state_head->state);
        }
        return RWKV_SUCCESS;
    }

    state_node* match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill);
    int register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits);
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