#include "backend.h"
#include "logger.h"
#include "commondef.h"
#include <cstring>
#include <queue>

namespace rwkvmobile {

state_node* execution_provider::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = state_root.get();
    size_t ids_compare_pos = 0;

    // find the deepest node that matches the input text
    while (node->children.size() > 0) {
        bool matched = false;
        for (auto &child : node->children) {
            if (ids_compare_pos + child->ids.size() < ids.size() && std::equal(ids.begin() + ids_compare_pos, ids.begin() + ids_compare_pos + child->ids.size(), child->ids.begin())) {
                node = child.get();
                node->activation_count++; // Increment matched child count
                ids_compare_pos += child->ids.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            break;
        }

        std::string debug_msg = "matched tokens:";
        for (auto id : node->ids) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGI("%s\n", debug_msg.c_str());
    }


    set_state(node->state);

    new_ids_to_prefill = std::vector<int>(ids.begin() + ids_compare_pos, ids.end());
    std::string debug_msg = "new tokens to prefill: ";
    for (auto id : new_ids_to_prefill) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());
    return node;
}

int execution_provider::register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits) {
    auto new_node = std::make_unique<state_node>();
    std::any new_state;
    if (new_node == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    get_state(new_state);
    new_node->ids = std::vector<int>(ids);
    new_node->logits.resize(vocab_size);
    memcpy(new_node->logits.data(), logits, vocab_size * sizeof(float));
    new_node->state = std::move(new_state);

    node->children.push_back(std::move(new_node));
    return RWKV_SUCCESS;
}

int execution_provider::register_batch_state_checkpoint(state_node* &node, std::vector<std::any> &states, const std::vector<std::vector<int>> &ids, const float *logits) {
    auto batch_size = states.size();
    if (ids.size() != batch_size) {
        LOGE("register_batch_state_checkpoint: ids size %d != batch size %d\n", ids.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (size_t i = 0; i < batch_size; i++) {
        auto new_node = std::make_unique<state_node>();
        new_node->ids = std::vector<int>(ids[i]);
        new_node->state = std::move(states[i]);
        new_node->logits.resize(vocab_size);
        memcpy(new_node->logits.data(), logits + i * vocab_size, vocab_size * sizeof(float));
        node->children.push_back(std::move(new_node));
    }
    return RWKV_SUCCESS;
}

}
