#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;
    runtime.set_sampler_params(model_id, 1.0, 1, 1.0);
    runtime.set_penalty_params(model_id, 0.0, 0.0, 0.0);

    std::cout << "Generating demo text..." << std::endl;

    std::string prompt = "The Eiffel tower is in the city of";
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.gen_completion(model_id, prompt, 200, 261, nullptr), "Failed to generate chat message");
    std::cout << runtime.get_response_buffer_content(model_id);

    std::cout << std::endl;

    std::cout << "Prefill speed: " << runtime.get_avg_prefill_speed(model_id) << " tokens/s" << std::endl;
    std::cout << "Decode speed: " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
