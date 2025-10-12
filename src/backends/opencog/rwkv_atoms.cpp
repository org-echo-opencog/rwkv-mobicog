#include "rwkv_atoms.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <map>
#include <unordered_map>

namespace rwkvmobile {
namespace opencog {

// Atom implementation
Atom::Atom(RWKVAtomType t, const std::string& n) : type(t), name(n) {}

// AtomSpace implementation
void AtomSpace::add_atom(std::shared_ptr<Atom> atom) {
    atoms.push_back(atom);
    name_index[atom->name] = atom;
}

std::shared_ptr<Atom> AtomSpace::get_atom(const std::string& name) {
    auto it = name_index.find(name);
    return it != name_index.end() ? it->second : nullptr;
}

// RWKVAtomFactory implementation
std::shared_ptr<Atom> RWKVAtomFactory::create_token_atom(int token_id, const std::string& token_text) {
    std::stringstream ss;
    ss << "token_" << token_id;
    auto atom = std::make_shared<Atom>(RWKVAtomType::RWKV_TOKEN_NODE, ss.str());
    atom->properties["token_id"] = static_cast<float>(token_id);
    atom->properties["frequency"] = 1.0f;
    return atom;
}

std::shared_ptr<Atom> RWKVAtomFactory::create_state_atom(int layer_id, const std::vector<float>& state_vector) {
    std::stringstream ss;
    ss << "state_layer_" << layer_id;
    auto atom = std::make_shared<Atom>(RWKVAtomType::RWKV_STATE_NODE, ss.str());
    atom->features = state_vector;
    atom->properties["layer_id"] = static_cast<float>(layer_id);
    atom->properties["state_norm"] = 0.0f;
    
    // Calculate state norm
    float norm = 0.0f;
    for (float val : state_vector) {
        norm += val * val;
    }
    atom->properties["state_norm"] = std::sqrt(norm);
    
    return atom;
}

std::shared_ptr<Atom> RWKVAtomFactory::create_attention_link(std::shared_ptr<Atom> from_token, std::shared_ptr<Atom> to_token, float weight) {
    std::stringstream ss;
    ss << "attention_" << from_token->name << "_to_" << to_token->name;
    auto atom = std::make_shared<Atom>(RWKVAtomType::RWKV_ATTENTION_LINK, ss.str());
    atom->links.push_back(from_token);
    atom->links.push_back(to_token);
    atom->properties["weight"] = weight;
    atom->properties["strength"] = std::tanh(weight); // Normalize strength
    return atom;
}

std::shared_ptr<Atom> RWKVAtomFactory::create_sequence_link(const std::vector<std::shared_ptr<Atom>>& tokens) {
    std::stringstream ss;
    ss << "sequence_";
    for (size_t i = 0; i < tokens.size() && i < 3; ++i) {
        ss << tokens[i]->name << "_";
    }
    ss << "len_" << tokens.size();
    
    auto atom = std::make_shared<Atom>(RWKVAtomType::RWKV_SEQUENCE_LINK, ss.str());
    atom->links = tokens;
    atom->properties["length"] = static_cast<float>(tokens.size());
    atom->properties["coherence"] = 1.0f; // Could compute actual coherence
    return atom;
}

std::shared_ptr<Atom> RWKVAtomFactory::create_prediction_link(std::shared_ptr<Atom> context, std::shared_ptr<Atom> predicted_token, float probability) {
    std::stringstream ss;
    ss << "prediction_" << context->name << "_predicts_" << predicted_token->name;
    auto atom = std::make_shared<Atom>(RWKVAtomType::RWKV_PREDICTION_LINK, ss.str());
    atom->links.push_back(context);
    atom->links.push_back(predicted_token);
    atom->properties["probability"] = probability;
    atom->properties["confidence"] = probability; // Could be more sophisticated
    return atom;
}

// RWKVReasoning implementation
RWKVReasoning::RWKVReasoning(AtomSpace* atomspace) : atomspace_(atomspace) {}

std::vector<std::shared_ptr<Atom>> RWKVReasoning::find_similar_sequences(const std::vector<int>& query_tokens) {
    std::vector<std::shared_ptr<Atom>> results;
    
    // Simple similarity search based on token overlap
    for (auto& atom : atomspace_->atoms) {
        if (atom->type == RWKVAtomType::RWKV_SEQUENCE_LINK) {
            int overlap = 0;
            for (auto& linked_atom : atom->links) {
                for (int token_id : query_tokens) {
                    if (linked_atom->properties.count("token_id") && 
                        static_cast<int>(linked_atom->properties["token_id"]) == token_id) {
                        overlap++;
                        break;
                    }
                }
            }
            
            float similarity = static_cast<float>(overlap) / std::max(query_tokens.size(), atom->links.size());
            if (similarity > 0.3f) { // Threshold for similarity
                results.push_back(atom);
            }
        }
    }
    
    return results;
}

std::vector<std::shared_ptr<Atom>> RWKVReasoning::get_attention_targets(std::shared_ptr<Atom> source_token) {
    std::vector<std::shared_ptr<Atom>> targets;
    
    for (auto& atom : atomspace_->atoms) {
        if (atom->type == RWKVAtomType::RWKV_ATTENTION_LINK && 
            !atom->links.empty() && atom->links[0] == source_token) {
            if (atom->links.size() > 1) {
                targets.push_back(atom->links[1]);
            }
        }
    }
    
    // Sort by attention weight
    std::sort(targets.begin(), targets.end(), [&](const std::shared_ptr<Atom>& a, const std::shared_ptr<Atom>& b) {
        float weight_a = 0.0f, weight_b = 0.0f;
        
        for (auto& link : atomspace_->atoms) {
            if (link->type == RWKVAtomType::RWKV_ATTENTION_LINK) {
                if (link->links.size() >= 2 && link->links[1] == a) {
                    weight_a = link->properties.count("weight") ? link->properties.at("weight") : 0.0f;
                }
                if (link->links.size() >= 2 && link->links[1] == b) {
                    weight_b = link->properties.count("weight") ? link->properties.at("weight") : 0.0f;
                }
            }
        }
        
        return weight_a > weight_b;
    });
    
    return targets;
}

std::shared_ptr<Atom> RWKVReasoning::build_context_representation(const std::vector<int>& token_sequence) {
    std::vector<std::shared_ptr<Atom>> token_atoms;
    
    // Create or retrieve token atoms
    for (int token_id : token_sequence) {
        std::stringstream ss;
        ss << "token_" << token_id;
        auto token_atom = atomspace_->get_atom(ss.str());
        if (!token_atom) {
            token_atom = RWKVAtomFactory::create_token_atom(token_id, ss.str());
            atomspace_->add_atom(token_atom);
        }
        token_atoms.push_back(token_atom);
    }
    
    // Create context as a sequence
    auto context = RWKVAtomFactory::create_sequence_link(token_atoms);
    atomspace_->add_atom(context);
    
    return context;
}

float RWKVReasoning::estimate_token_probability(std::shared_ptr<Atom> context, int candidate_token) {
    // Simple probability estimation based on existing patterns
    float probability = 0.1f; // Base probability
    
    std::stringstream ss;
    ss << "token_" << candidate_token;
    auto candidate_atom = atomspace_->get_atom(ss.str());
    
    if (candidate_atom) {
        // Look for existing prediction links
        for (auto& atom : atomspace_->atoms) {
            if (atom->type == RWKVAtomType::RWKV_PREDICTION_LINK &&
                atom->links.size() >= 2 &&
                atom->links[0] == context &&
                atom->links[1] == candidate_atom) {
                probability = std::max(probability, atom->properties.count("probability") ? 
                                     atom->properties.at("probability") : 0.1f);
            }
        }
        
        // Factor in token frequency
        if (candidate_atom->properties.count("frequency")) {
            probability *= (1.0f + candidate_atom->properties.at("frequency") * 0.1f);
        }
    }
    
    return std::min(probability, 1.0f);
}

// RWKVCognitiveGraph implementation
RWKVCognitiveGraph::RWKVCognitiveGraph(AtomSpace* atomspace) 
    : atomspace_(atomspace), reasoning_(std::make_unique<RWKVReasoning>(atomspace)) {}

void RWKVCognitiveGraph::build_semantic_network(const std::vector<std::vector<int>>& token_sequences) {
    std::unordered_map<int, int> token_frequencies;
    std::map<std::pair<int, int>, int> cooccurrence;
    
    // Analyze token patterns
    for (const auto& sequence : token_sequences) {
        for (size_t i = 0; i < sequence.size(); ++i) {
            token_frequencies[sequence[i]]++;
            
            // Build co-occurrence patterns
            for (size_t j = i + 1; j < std::min(i + 5, sequence.size()); ++j) {
                auto pair = std::make_pair(sequence[i], sequence[j]);
                cooccurrence[pair]++;
            }
        }
    }
    
    // Create token atoms with frequency information
    for (const auto& [token_id, frequency] : token_frequencies) {
        std::stringstream ss;
        ss << "token_" << token_id;
        auto token_atom = RWKVAtomFactory::create_token_atom(token_id, ss.str());
        token_atom->properties["frequency"] = static_cast<float>(frequency);
        atomspace_->add_atom(token_atom);
    }
    
    // Create attention links based on co-occurrence
    for (const auto& [pair, count] : cooccurrence) {
        auto from_atom = atomspace_->get_atom("token_" + std::to_string(pair.first));
        auto to_atom = atomspace_->get_atom("token_" + std::to_string(pair.second));
        
        if (from_atom && to_atom && count > 2) { // Threshold for creating links
            float weight = std::log(1.0f + count);
            auto attention_link = RWKVAtomFactory::create_attention_link(from_atom, to_atom, weight);
            atomspace_->add_atom(attention_link);
        }
    }
}

std::vector<std::shared_ptr<Atom>> RWKVCognitiveGraph::extract_concepts(const std::vector<int>& tokens) {
    std::vector<std::shared_ptr<Atom>> concepts;
    
    // Simple concept extraction based on token clusters
    auto context = reasoning_->build_context_representation(tokens);
    auto similar_sequences = reasoning_->find_similar_sequences(tokens);
    
    for (auto& sequence : similar_sequences) {
        if (sequence->properties.count("coherence") && 
            sequence->properties.at("coherence") > 0.7f) {
            concepts.push_back(sequence);
        }
    }
    
    return concepts;
}

std::vector<int> RWKVCognitiveGraph::generate_goal_oriented_sequence(std::shared_ptr<Atom> goal, int max_length) {
    std::vector<int> sequence;
    
    // Simple goal-oriented generation (would be more sophisticated in practice)
    // For now, we'll generate based on attention patterns from the goal
    if (goal && goal->type == RWKVAtomType::RWKV_TOKEN_NODE && 
        goal->properties.count("token_id")) {
        
        int current_token = static_cast<int>(goal->properties.at("token_id"));
        sequence.push_back(current_token);
        
        for (int i = 1; i < max_length; ++i) {
            auto current_atom = atomspace_->get_atom("token_" + std::to_string(current_token));
            if (!current_atom) break;
            
            auto targets = reasoning_->get_attention_targets(current_atom);
            if (targets.empty()) break;
            
            // Select next token based on attention strength
            auto next_atom = targets[0];
            if (next_atom->properties.count("token_id")) {
                current_token = static_cast<int>(next_atom->properties.at("token_id"));
                sequence.push_back(current_token);
            } else {
                break;
            }
        }
    }
    
    return sequence;
}

void RWKVCognitiveGraph::consolidate_episodic_memory(const std::vector<std::shared_ptr<Atom>>& episodes) {
    // Simple memory consolidation - merge similar episodes and strengthen patterns
    std::unordered_map<std::string, std::vector<std::shared_ptr<Atom>>> episode_groups;
    
    for (auto& episode : episodes) {
        if (episode->type == RWKVAtomType::RWKV_SEQUENCE_LINK) {
            // Group by sequence length and first token
            std::string key = std::to_string(episode->links.size());
            if (!episode->links.empty() && episode->links[0]->properties.count("token_id")) {
                key += "_" + std::to_string(static_cast<int>(episode->links[0]->properties.at("token_id")));
            }
            episode_groups[key].push_back(episode);
        }
    }
    
    // Strengthen patterns that appear frequently
    for (const auto& [key, group] : episode_groups) {
        if (group.size() > 3) { // Threshold for consolidation
            for (auto& episode : group) {
                if (episode->properties.count("coherence")) {
                    episode->properties["coherence"] *= 1.1f; // Strengthen
                    episode->properties["coherence"] = std::min(episode->properties["coherence"], 1.0f);
                }
            }
        }
    }
}

} // namespace opencog
} // namespace rwkvmobile