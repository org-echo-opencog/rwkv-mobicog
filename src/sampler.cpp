#include "sampler.h"

namespace rwkvmobile {

NucleusSampler::NucleusSampler() {
    _generator.seed(std::random_device()());
}

int NucleusSampler::sample(const float* logits, const size_t size) {
    return sample(logits, size, _temperature, _top_k, _top_p);
}

int NucleusSampler::sample(const float* logits, const size_t size, float temperature, int top_k, float top_p) {
    if (logits == nullptr) {
        return 0;
    }
    temperature = std::clamp(temperature, 0.1f, 5.f);
    if (top_k >= size || top_k == 0)
        top_k = size;

    if (top_k == 1 || fabs(top_p - 0.f) < 1e-4)
        return std::max_element(logits, logits + size) - logits;

    if (index_buffer.size() != size) {
        index_buffer.resize(size);
    }
    for (int i = 0; i < size; i++) {
        index_buffer[i] = i;
    }

    if (top_k != size)
        std::nth_element(index_buffer.begin(), index_buffer.begin() + top_k,
                index_buffer.end(),
                [&](int i, int j) { return logits[i] > logits[j]; });
    std::sort(index_buffer.begin(), index_buffer.begin() + top_k,
            [&](int i, int j) { return logits[i] > logits[j]; });

    int len = top_k;
    if (probs_buffer.size() != len) {
        probs_buffer.resize(len);
    }

    // softmax
    float sum = 0;
    for (int i = 0; i < len; i++) {
        probs_buffer[i] = std::exp(logits[index_buffer[i]] / temperature);
        sum += probs_buffer[i];
    }

    // top-p
    float cumsum = 0;
    for (int i = 0; i < len; i++) {
        probs_buffer[i] /= sum;
        cumsum += probs_buffer[i];
        if (cumsum >= top_p) {
            len = i + 1;
            break;
        }
    }

    // random choice
    float random_value = cumsum * (_generator() - _generator.min()) /
                        (_generator.max() - _generator.min());
    
    int ret = -1;
    cumsum = 0;
    for (int i = 0; i < len; i++) {
        cumsum += probs_buffer[i];
        if (cumsum >= random_value) {
            ret = index_buffer[i];
            break;
        }
    }
    return ret;
}

void NucleusSampler::set_seed(int seed) {
    _generator.seed(seed);
}

void NucleusSampler::update_occurences(int token) {
    _occurences[token]++;
}

void NucleusSampler::apply_penalties(float * logits, const size_t size, std::map<int, float> occurences, std::vector<int> token_banned, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (!logits) {
        return;
    }
    for (auto &[id, occurence] : occurences) {
        if (id >= size) {
            continue;
        }
        logits[id] -=
            frequency_penalty * occurence + presence_penalty;
        occurences[id] *= penalty_decay;
    }

    for (auto &token : token_banned) {
        if (token >= size) {
            continue;
        }
        logits[token] = -INFINITY;
    }
}

void NucleusSampler::apply_penalties(float * logits, const size_t size) {
    if (_presence_penalty > 0.0f && _frequency_penalty > 0.0f && _penalty_decay > 0.0f) {
        apply_penalties(logits, size, _occurences, _token_banned, _presence_penalty, _frequency_penalty, _penalty_decay);
    }
}

}