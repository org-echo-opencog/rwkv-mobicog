#ifndef LLAMA_CPP_BACKEND_H
#define LLAMA_CPP_BACKEND_H

#include "backend.h"
#include "llama.h"

namespace rwkvmobile {

class llama_cpp_backend : public execution_provider {
public:
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, float *& logits) override;
    void free_logits_if_allocated(float *& logits) override {
        return;
    };
    bool is_available() override;
    int clear_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;
private:
    llama_model * model;
    llama_context * ctx;
};

}

#endif
