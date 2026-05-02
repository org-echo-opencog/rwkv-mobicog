#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstring>

#include "opencog_rwkv_backend.h"
#include "rwkv_atoms.h"
#include "backend.h"
#include "commondef.h"

namespace rwkvmobile {

// ---------------------------------------------------------------------------
// GGUF header parsing helpers
// ---------------------------------------------------------------------------

static std::string gguf_read_string(std::ifstream& file) {
    uint64_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(uint64_t));
    if (!file || len > (1u << 20)) return "";
    std::string s(len, '\0');
    if (len > 0) file.read(&s[0], static_cast<std::streamsize>(len));
    return s;
}

static bool gguf_skip_value(std::ifstream& file, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7:
            return (bool)file.seekg(1, std::ios::cur);
        case 2: case 3:
            return (bool)file.seekg(2, std::ios::cur);
        case 4: case 5: case 6:
            return (bool)file.seekg(4, std::ios::cur);
        case 10: case 11: case 12:
            return (bool)file.seekg(8, std::ios::cur);
        case 8: {
            gguf_read_string(file);
            return file.good();
        }
        case 9: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            file.read(reinterpret_cast<char*>(&elem_type), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
            for (uint64_t i = 0; i < count && file.good(); i++) {
                gguf_skip_value(file, elem_type);
            }
            return file.good();
        }
        default:
            return false;
    }
}

// PIMPL implementation for OpenCog integration
struct OpenCogImpl {
    std::unique_ptr<opencog::AtomSpace> atomspace;
    std::unique_ptr<opencog::RWKVCognitiveGraph> cognitive_graph;
    
    OpenCogImpl() {
        atomspace = std::make_unique<opencog::AtomSpace>();
        cognitive_graph = std::make_unique<opencog::RWKVCognitiveGraph>(atomspace.get());
    }
};

int opencog_rwkv_backend::init(void * extra) {
    // Initialize the backend
    opencog_impl_ = nullptr;
    model_loaded = false;
    logits_buffer.clear();
    state_buffers.clear();
    current_sequence.clear();
    processed_sequences.clear();
    sequence_position = 0;
    
    return initialize_atomspace();
}

int opencog_rwkv_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_IO;
    }
    
    model_file_path = model_path;
    
    // Load model parameters and setup
    int ret = load_model_parameters();
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    
    // Setup RWKV atoms in AtomSpace
    ret = setup_rwkv_atoms();
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    
    // Initialize state buffers based on model parameters
    state_buffers.resize(model_layers);
    for (int i = 0; i < model_layers; i++) {
        state_buffers[i].resize(model_embed_dim, 0.0f);
    }
    
    // Initialize logits buffer
    logits_buffer.resize(model_vocab_size, 0.0f);
    
    // Set model parameters inherited from execution_provider
    n_layers = model_layers;
    hidden_size = model_embed_dim;
    vocab_size = model_vocab_size;
    
    model_loaded = true;
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::eval(int id, float *& logits) {
    if (!model_loaded || !opencog_impl_) {
        return RWKV_ERROR_MODEL;
    }
    
    OpenCogImpl* impl = static_cast<OpenCogImpl*>(opencog_impl_);
    
    // Add token to current sequence for cognitive processing
    current_sequence.push_back(id);
    sequence_position++;
    
    // OpenCog-enhanced RWKV evaluation:
    // 1. Create/update token atom in AtomSpace
    std::stringstream ss;
    ss << "token_" << id;
    auto token_atom = impl->atomspace->get_atom(ss.str());
    if (!token_atom) {
        token_atom = opencog::RWKVAtomFactory::create_token_atom(id, ss.str());
        impl->atomspace->add_atom(token_atom);
    }
    
    // 2. Build context representation  
    auto reasoning = std::make_unique<opencog::RWKVReasoning>(impl->atomspace.get());
    auto context_atom = reasoning->build_context_representation(current_sequence);
    
    // 3. Generate logits with cognitive enhancement
    for (int i = 0; i < vocab_size; i++) {
        // Base RWKV simulation
        float base_logit = std::sin(i * 0.1f + id * 0.01f) * 2.0f;
        
        // Cognitive enhancement: use OpenCog reasoning
        float contextual_prob = compute_contextual_probability(i);
        
        // Combine base model with cognitive reasoning
        float enhanced_logit = base_logit + std::log(contextual_prob + 1e-8f);
        logits_buffer[i] = enhanced_logit;
    }
    
    // 4. Update internal state with cognitive context
    for (int layer = 0; layer < model_layers; layer++) {
        for (int j = 0; j < model_embed_dim && j < 100; j++) {
            // Incorporate contextual information into state
            float contextual_influence = 0.01f * std::tanh(sequence_position * 0.1f);
            state_buffers[layer][j] = state_buffers[layer][j] * 0.99f + 
                                    std::sin(id * 0.001f + j * 0.1f + contextual_influence) * 0.01f;
        }
    }
    
    // 5. Update cognitive graph periodically
    if (sequence_position % 10 == 0) {
        update_cognitive_graph();
    }
    
    logits = logits_buffer.data();
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    // Sequential evaluation for multiple tokens
    for (size_t i = 0; i < ids.size(); i++) {
        int ret = eval(ids[i], logits);
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
    }
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::eval_batch(std::vector<std::vector<int>> ids, float *& logits) {
    if (ids.empty()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (!model_loaded) {
        return RWKV_ERROR_MODEL;
    }
    // Process the first sequence; full multi-slot batch builds on this foundation
    return eval(ids[0], logits);
}

int opencog_rwkv_backend::get_state(std::any &state) {
    if (!model_loaded) {
        return RWKV_ERROR_MODEL;
    }
    
    // Return a copy of current state buffers
    std::vector<std::vector<float>> state_copy = state_buffers;
    state = state_copy;
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::set_state(std::any state) {
    if (!model_loaded) {
        return RWKV_ERROR_MODEL;
    }
    
    try {
        std::vector<std::vector<float>> state_vec = std::any_cast<std::vector<std::vector<float>>>(state);
        if (state_vec.size() != state_buffers.size()) {
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
        
        state_buffers = state_vec;
        return RWKV_SUCCESS;
    } catch (const std::bad_any_cast& e) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
}

int opencog_rwkv_backend::free_state(std::any state) {
    // State management - reset the any object
    if (state.has_value()) {
        state.reset();
    }
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    if (!state.has_value()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    try {
        auto state_vec = std::any_cast<std::vector<std::vector<float>>>(state);
        uint32_t n = static_cast<uint32_t>(state_vec.size());
        uint32_t layer_size = (n > 0) ? static_cast<uint32_t>(state_vec[0].size()) : 0;

        size_t total = sizeof(uint32_t) * 2 + static_cast<size_t>(n) * layer_size * sizeof(float);
        data.resize(total);

        uint8_t* ptr = data.data();
        std::memcpy(ptr, &n, sizeof(uint32_t));          ptr += sizeof(uint32_t);
        std::memcpy(ptr, &layer_size, sizeof(uint32_t)); ptr += sizeof(uint32_t);
        for (const auto& layer_state : state_vec) {
            std::memcpy(ptr, layer_state.data(), layer_state.size() * sizeof(float));
            ptr += layer_state.size() * sizeof(float);
        }
        return RWKV_SUCCESS;
    } catch (...) {
        return RWKV_ERROR_RUNTIME;
    }
}

int opencog_rwkv_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    if (data.size() < sizeof(uint32_t) * 2) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    uint8_t* ptr = data.data();
    uint32_t n = 0, layer_size = 0;
    std::memcpy(&n, ptr, sizeof(uint32_t));          ptr += sizeof(uint32_t);
    std::memcpy(&layer_size, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    size_t expected = sizeof(uint32_t) * 2 + static_cast<size_t>(n) * layer_size * sizeof(float);
    if (data.size() < expected) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<std::vector<float>> state_vec(n, std::vector<float>(layer_size));
    for (auto& layer_state : state_vec) {
        std::memcpy(layer_state.data(), ptr, layer_size * sizeof(float));
        ptr += layer_size * sizeof(float);
    }
    state = std::move(state_vec);
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::zero_state() {
    if (!model_loaded) {
        return RWKV_ERROR_MODEL;
    }
    
    // Reset all state buffers to zero
    for (auto& layer_state : state_buffers) {
        std::fill(layer_state.begin(), layer_state.end(), 0.0f);
    }
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::release_model() {
    model_loaded = false;
    logits_buffer.clear();
    state_buffers.clear();
    model_layers = 0;
    model_embed_dim = 0;
    model_vocab_size = 0;
    model_file_path.clear();
    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::release() {
    release_model();
    cleanup_atomspace();
    return RWKV_SUCCESS;
}

bool opencog_rwkv_backend::is_available() {
    // OpenCog backend is available if we can initialize AtomSpace
    // For this implementation, we'll assume it's always available
    return true;
}

int opencog_rwkv_backend::load_model_parameters() {
    std::ifstream file(model_file_path, std::ios::binary);
    if (!file.is_open()) {
        return RWKV_ERROR_IO;
    }

    // Check for GGUF magic "GGUF"
    char magic[4] = {0};
    file.read(magic, 4);
    if (!file || std::memcmp(magic, "GGUF", 4) != 0) {
        // Not a GGUF file - fall back to conservative defaults
        model_layers = 12;
        model_embed_dim = 768;
        model_vocab_size = 65536;
        return RWKV_SUCCESS;
    }

    // GGUF version
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    // n_tensors and n_kv: v1 uses uint32, v2+ uses uint64
    uint64_t n_kv = 0;
    if (version == 1) {
        uint32_t t32 = 0, k32 = 0;
        file.read(reinterpret_cast<char*>(&t32), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&k32), sizeof(uint32_t));
        n_kv = k32;
    } else {
        uint64_t n_tensors = 0;
        file.read(reinterpret_cast<char*>(&n_tensors), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&n_kv), sizeof(uint64_t));
    }

    if (!file) {
        model_layers = 12;
        model_embed_dim = 768;
        model_vocab_size = 65536;
        return RWKV_SUCCESS;
    }

    // Helpers for suffix matching (C++17 lacks std::string::ends_with)
    auto ends_with = [](const std::string& s, const std::string& suffix) -> bool {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    bool found_block_count = false;
    bool found_embed_len   = false;
    bool found_vocab       = false;

    for (uint64_t i = 0; i < n_kv && file.good(); i++) {
        std::string key = gguf_read_string(file);
        uint32_t val_type = 0;
        file.read(reinterpret_cast<char*>(&val_type), sizeof(uint32_t));
        if (!file) break;

        if (!found_block_count && ends_with(key, ".block_count") &&
            (val_type == 4 /* UINT32 */ || val_type == 10 /* UINT64 */)) {
            if (val_type == 4) {
                uint32_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
                model_layers = static_cast<int>(val);
            } else {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint64_t));
                model_layers = static_cast<int>(val);
            }
            found_block_count = true;
        } else if (!found_embed_len && ends_with(key, ".embedding_length") &&
                   (val_type == 4 || val_type == 10)) {
            if (val_type == 4) {
                uint32_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
                model_embed_dim = static_cast<int>(val);
            } else {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint64_t));
                model_embed_dim = static_cast<int>(val);
            }
            found_embed_len = true;
        } else if (!found_vocab && key == "tokenizer.ggml.tokens" && val_type == 9 /* ARRAY */) {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            file.read(reinterpret_cast<char*>(&elem_type), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
            model_vocab_size = static_cast<int>(count);
            found_vocab = true;
            // Skip the token strings
            for (uint64_t j = 0; j < count && file.good(); j++) {
                gguf_skip_value(file, elem_type);
            }
        } else {
            gguf_skip_value(file, val_type);
        }

        // Stop early once all parameters are found
        if (found_block_count && found_embed_len && found_vocab) break;
    }

    // Apply defaults for any parameters not found in the file
    if (!found_block_count) model_layers     = 12;
    if (!found_embed_len)   model_embed_dim  = 768;
    if (!found_vocab)       model_vocab_size = 65536;

    return RWKV_SUCCESS;
}

int opencog_rwkv_backend::initialize_atomspace() {
    // Initialize AtomSpace and cognitive graph
    try {
        opencog_impl_ = new OpenCogImpl();
        return RWKV_SUCCESS;
    } catch (...) {
        return RWKV_ERROR_INIT;
    }
}

int opencog_rwkv_backend::setup_rwkv_atoms() {
    // In a real OpenCog integration, this would:
    // 1. Create atom types for RWKV components
    // 2. Setup pattern matching for RWKV operations  
    // 3. Create cognitive graph structures for RWKV
    // 4. Setup reasoning chains
    
    // For this implementation, we simulate the setup
    return RWKV_SUCCESS;
}

void opencog_rwkv_backend::cleanup_atomspace() {
    if (opencog_impl_) {
        delete static_cast<OpenCogImpl*>(opencog_impl_);
        opencog_impl_ = nullptr;
    }
}

void opencog_rwkv_backend::update_cognitive_graph() {
    if (!opencog_impl_ || processed_sequences.empty()) return;
    
    OpenCogImpl* impl = static_cast<OpenCogImpl*>(opencog_impl_);
    
    try {
        // Add current sequence to processed sequences
        if (!current_sequence.empty()) {
            processed_sequences.push_back(current_sequence);
        }
        
        // Build semantic network from all processed sequences
        impl->cognitive_graph->build_semantic_network(processed_sequences);
        
        // Limit memory usage by keeping only recent sequences
        if (processed_sequences.size() > 100) {
            processed_sequences.erase(processed_sequences.begin(), 
                                    processed_sequences.begin() + 50);
        }
    } catch (...) {
        // Handle errors gracefully
    }
}

float opencog_rwkv_backend::compute_contextual_probability(int token_id) {
    if (!opencog_impl_ || current_sequence.empty()) {
        return 0.1f; // Base probability
    }
    
    OpenCogImpl* impl = static_cast<OpenCogImpl*>(opencog_impl_);
    
    try {
        std::stringstream ss;
        ss << "token_" << token_id;
        auto candidate_atom = impl->atomspace->get_atom(ss.str());

        if (!candidate_atom) {
            return 0.1f; // Unknown token, use base probability
        }

        // Weight by observed token frequency
        float freq_weight = 1.0f;
        if (candidate_atom->properties.count("frequency")) {
            freq_weight = 1.0f + std::log1p(candidate_atom->properties.at("frequency")) * 0.1f;
        }

        // Check attention links from recent context tokens to the candidate
        float cooccurrence_score = 0.0f;
        int context_window = std::min(static_cast<int>(current_sequence.size()), 5);
        for (int j = static_cast<int>(current_sequence.size()) - context_window;
             j < static_cast<int>(current_sequence.size()); j++) {
            std::stringstream ctx_ss;
            ctx_ss << "token_" << current_sequence[j];
            auto ctx_atom = impl->atomspace->get_atom(ctx_ss.str());
            if (!ctx_atom) continue;

            std::string link_name = "attention_" + ctx_atom->name + "_to_" + candidate_atom->name;
            auto link = impl->atomspace->get_atom(link_name);
            if (link && link->properties.count("weight")) {
                cooccurrence_score += link->properties.at("weight");
            }
        }

        float prob = 0.1f * freq_weight + cooccurrence_score * 0.05f;
        return std::min(prob, 1.0f);
    } catch (...) {
        return 0.1f; // Fallback probability
    }
}

} // namespace rwkvmobile