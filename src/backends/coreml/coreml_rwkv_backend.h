#ifndef COREML_RWKV_BACKEND_H
#define COREML_RWKV_BACKEND_H

#include "backend.h"
#include "rwkv-coreml.h"

namespace rwkvmobile {

class coreml_rwkv_backend : public execution_provider {
public:
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    void free_logits_if_allocated(float *& logits) override {
        return;
    };
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    int release() override;

private:
    rwkv_coreml_context * ctx;
};

}

#endif
