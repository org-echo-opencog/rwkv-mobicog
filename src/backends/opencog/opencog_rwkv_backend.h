#ifndef OPENCOG_RWKV_BACKEND_H
#define OPENCOG_RWKV_BACKEND_H

#include "backend.h"

namespace rwkvmobile {

class opencog_rwkv_backend : public execution_provider {
public:
    ~opencog_rwkv_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    int release() override;

private:
    // OpenCog AtomSpace integration members
    void* atomspace;  // Will cast to OpenCog's AtomSpace* when needed
    std::vector<float> logits_buffer;
    std::vector<std::vector<float>> state_buffers;
    bool model_loaded = false;
    std::string model_file_path;
    
    // RWKV model parameters
    int model_layers = 0;
    int model_embed_dim = 0;
    int model_vocab_size = 0;
    
    // Internal helper methods
    int load_model_parameters();
    int initialize_atomspace();
    int setup_rwkv_atoms();
    void cleanup_atomspace();
};

}

#endif