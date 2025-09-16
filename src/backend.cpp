#include "backend.h"
#include "logger.h"
#include "commondef.h"
#include <cstring>
#include <queue>
#include <functional>

namespace rwkvmobile {

state_node* execution_provider::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = state_root.get();
    size_t ids_compare_pos = 0;

    // find the deepest node that matches the input text
    while (node->children.size() > 0) {
        bool matched = false;
        for (auto &child : node->children) {
            if (ids_compare_pos + child->ids.size() < ids.size()) {
                LOGD("ids_compare_pos = %d", ids_compare_pos);
                std::string debug_msg = "child->ids = ";
                for (auto id : child->ids) {
                    debug_msg += std::to_string(id) + " ";
                }
                LOGD("%s\n", debug_msg.c_str());
                debug_msg = "ids = ";
                for (int i = ids_compare_pos; i < ids_compare_pos + child->ids.size(); i++) {
                    debug_msg += std::to_string(ids[i]) + " ";
                }
                LOGD("%s\n", debug_msg.c_str());
            }
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
    return node;
}

int execution_provider::register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits) {
    for (auto &child : node->children) {
        if (child->ids.size() == ids.size() && std::equal(child->ids.begin(), child->ids.end(), ids.begin())) {
            child->activation_count++;
            node = child.get();
            return RWKV_SUCCESS;
        }
    }

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
    new_node->activation_count = node->activation_count;

    std::string debug_msg = "register_state_checkpoint: new node ids: ";
    for (auto id : new_node->ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    node->children.push_back(std::move(new_node));
    node = node->children.back().get();
    return RWKV_SUCCESS;
}

int execution_provider::register_batch_state_checkpoint(std::vector<state_node*> &nodes, std::vector<std::any> &states, const std::vector<std::vector<int>> &ids, const float *logits) {
    auto batch_size = states.size();
    if (ids.size() != batch_size) {
        LOGE("register_batch_state_checkpoint: ids size %d != batch size %d\n", ids.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (size_t i = 0; i < batch_size; i++) {
        for (auto &child : nodes[i]->children) {
            if (child->ids.size() == ids[i].size() && std::equal(child->ids.begin(), child->ids.end(), ids[i].begin())) {
                child->activation_count++;
                continue;
            }
        }
        auto new_node = std::make_unique<state_node>();
        new_node->ids = std::vector<int>(ids[i]);
        new_node->state = std::move(states[i]);
        new_node->logits.resize(vocab_size);
        memcpy(new_node->logits.data(), logits + i * vocab_size, vocab_size * sizeof(float));
        new_node->activation_count = nodes[i]->activation_count;

        nodes[i]->children.push_back(std::move(new_node));
        nodes[i] = nodes[i]->children.back().get();
    }
    return RWKV_SUCCESS;
}

void execution_provider::cleanup_state_tree() {
    if (!state_root) {
        return;
    }

    std::function<void(state_node*)> cleanup_node = [&](state_node* node) {
        if (!node) {
            return;
        }

        for (auto it = node->children.begin(); it != node->children.end();) {
            cleanup_node(it->get());

            if (!(*it)->is_constant && (*it)->activation_count <= 0) {
                it = node->children.erase(it);
            } else {
                if (!(*it)->is_constant) {
                    (*it)->activation_count--;
                }
                ++it;
            }
        }
    };

    cleanup_node(state_root.get());
}

}
