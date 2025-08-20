#ifndef WEB_RWKV_BACKEND_H
#define WEB_RWKV_BACKEND_H

#include "backend.h"
#include "web_rwkv_ffi.h"
#include <memory>

namespace rwkvmobile {

class web_rwkv_state {
public:
    web_rwkv_state(StateRaw raw) : raw(raw) {
    }

    ~web_rwkv_state() {
        if (raw.state && raw.len > 0) {
            ::free_state(raw);
        }
    }


    web_rwkv_state(const web_rwkv_state&) = delete;
    web_rwkv_state& operator=(const web_rwkv_state&) = delete;

    web_rwkv_state(web_rwkv_state&& other) noexcept
        : raw(other.raw) {
        other.raw.state = nullptr;
        other.raw.len = 0;
    }

    web_rwkv_state& operator=(web_rwkv_state&& other) noexcept {
        if (this != &other) {
            if (raw.state && raw.len > 0) {
                ::free_state(raw);
            }
            raw = other.raw;
            other.raw.state = nullptr;
            other.raw.len = 0;
        }
        return *this;
    }

    StateRaw raw;
};

class web_rwkv_backend : public execution_provider {
public:
    ~web_rwkv_backend() {
        if (state_head) {
            state_head->delete_after();
            state_head = nullptr;
        }
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    void free_logits_if_allocated(float *& logits) override;
    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;
private:
    int logits_len_from_backend = 0;
};

}

#endif
