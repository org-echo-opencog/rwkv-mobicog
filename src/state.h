#ifndef STATE_H
#define STATE_H

#include <any>
#include <vector>

namespace rwkvmobile {

class state_node {
public:
    std::any state;
    std::vector<int> ids;
    std::vector<float> last_logits;
    struct state_node * next = nullptr;

    state_node() {
        state = nullptr;
        ids = {};
        last_logits = {};
        next = nullptr;
    }

    ~state_node() {
        if (state.has_value()) {
            state.reset();
        }
    }

    inline void delete_after() {
        while (next) {
            auto tmp = next->next;
            // free_state(next->state);
            next->state.reset();
            delete next;
            next = tmp;
        }
    }
};

}

#endif