#include "backend.h"
#include "logger.h"
#include "commondef.h"
#include <cstring>

namespace rwkvmobile {

state_node* execution_provider::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = state_head;
    size_t compare_pos = 0;

    // find the last node that matches the input text
    while (node->next) {

        if (compare_pos + node->next->ids.size() > ids.size() || !std::equal(ids.begin() + compare_pos, ids.begin() + compare_pos + node->next->ids.size(), node->next->ids.begin())) {
            // the text will diverge at next node
            break;
        }
        std::string debug_msg = "matched tokens:";
        for (auto id : node->next->ids) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGI("%s\n", debug_msg.c_str());
        compare_pos += node->next->ids.size();
        node = node->next;
    }

    set_state(node->state);
    node->delete_after();

    new_ids_to_prefill = std::vector<int>(ids.begin() + compare_pos, ids.end());
    std::string debug_msg = "new tokens to prefill: ";
    for (auto id : new_ids_to_prefill) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());
    return node;
}
    
int execution_provider::register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits) {
    node->next = new state_node;
    if (node->next == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    node = node->next;
    node->ids = std::vector<int>(ids);
    get_state(node->state);
    node->last_logits.resize(vocab_size);
    memcpy(node->last_logits.data(), logits, vocab_size * sizeof(float));
    return RWKV_SUCCESS;
}

}