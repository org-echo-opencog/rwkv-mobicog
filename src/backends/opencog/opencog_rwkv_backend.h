#ifndef OPENCOG_RWKV_BACKEND_H
#define OPENCOG_RWKV_BACKEND_H

#include "backend.h"

namespace rwkvmobile {
namespace opencog {
    struct AtomSpace;
    class RWKVCognitiveGraph;
}
}

namespace rwkvmobile {

class opencog_rwkv_backend : public execution_provider {
public:
    opencog_rwkv_backend() : opencog_impl_(nullptr), model_loaded(false), model_layers(0), model_embed_dim(0), model_vocab_size(0) {}
    
    ~opencog_rwkv_backend() {
        try {
            release_model();
            release();
        } catch (...) {
            // Ensure destructor doesn't throw
        }
    }
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    int eval_batch(std::vector<std::vector<int>> ids, float *& logits) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    int release() override;
    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;

private:
    // OpenCog integration (using void* to avoid incomplete type issues)
    void* opencog_impl_;  // Points to OpenCogImpl struct
    
    std::vector<float> logits_buffer;
    std::vector<std::vector<float>> state_buffers;
    std::vector<std::vector<int>> processed_sequences;
    bool model_loaded = false;
    std::string model_file_path;
    
    // RWKV model parameters
    int model_layers = 0;
    int model_embed_dim = 0;
    int model_vocab_size = 0;
    
    // Processing context
    std::vector<int> current_sequence;
    int sequence_position = 0;
    
    // Internal helper methods
    int load_model_parameters();
    int initialize_atomspace();
    int setup_rwkv_atoms();
    void cleanup_atomspace();
    void update_cognitive_graph();
    float compute_contextual_probability(int token_id);
};

}

#endif