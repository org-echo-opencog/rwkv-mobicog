#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <random>
#include <algorithm>
#include <vector>
#include <map>

namespace rwkvmobile {

class NucleusSampler {
public:
    NucleusSampler();

    void apply_penalties(float * logits, const size_t size, std::map<int, float> occurences, std::vector<int> token_banned, float presence_penalty, float frequency_penalty, float penalty_decay);

    void apply_penalties(float * logits, const size_t size);

    void clear_occurences() { _occurences.clear(); }

    void update_occurences(int token);

    int sample(const float* logits, const size_t size);

    int sample(const float* logits, const size_t size, float temperature, int top_k, float top_p);

    void set_seed(int seed);

    void set_temperature(float temperature) { _temperature = temperature; }
    void set_top_k(int top_k) { _top_k = top_k; }
    void set_top_p(float top_p) { _top_p = top_p; }
    void set_presence_penalty(float presence_penalty) { _presence_penalty = presence_penalty; }
    void set_frequency_penalty(float frequency_penalty) { _frequency_penalty = frequency_penalty; }
    void set_penalty_decay(float penalty_decay) { _penalty_decay = penalty_decay; }
    void set_token_banned(std::vector<int> token_banned) { _token_banned = token_banned; }

    float get_temperature() { return _temperature; }
    int get_top_k() { return _top_k; }
    float get_top_p() { return _top_p; }
    float get_presence_penalty() { return _presence_penalty; }
    float get_frequency_penalty() { return _frequency_penalty; }
    float get_penalty_decay() { return _penalty_decay; }
    std::vector<int> get_token_banned() { return _token_banned; }

private:
    std::minstd_rand0 _generator;

    std::vector<float> probs_buffer;
    std::vector<int> index_buffer;

    std::vector<int> _token_banned;

    float _temperature = 1.0f;
    int _top_k = 128;
    float _top_p = 0.5f;
    float _presence_penalty = 0.5;
    float _frequency_penalty = 0.5;
    float _penalty_decay = 0.996;

    std::map<int, float> _occurences;
};

}
#endif