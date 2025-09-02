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
    std::vector<struct state_node *> batch_states;

    state_node() {
    }

    ~state_node() {
        if (state.has_value()) {
            state.reset();
        }
        for (auto s : batch_states) {
            s->state.reset();
        }
        batch_states.clear();
    }

    inline void delete_after() {
        while (next) {
            auto tmp = next->next;
            next->state.reset();
            delete next;
            next = tmp;
        }
    }
};

}

#endif