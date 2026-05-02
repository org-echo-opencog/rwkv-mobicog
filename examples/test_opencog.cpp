#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_EXIT(x, msg) \
    if ((x) != rwkvmobile::RWKV_SUCCESS) { std::cerr << (msg) << std::endl; return 1; }

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file>" << std::endl;
        std::cerr << "  The model_file can be a GGUF file or any supported RWKV format." << std::endl;
        return 1;
    }

    const char *vocab_path = argv[1];
    const char *model_path = argv[2];

    rwkvmobile::runtime rt;

    int model_id = rt.load_model(model_path, "opencog", vocab_path, nullptr);
    if (model_id < 0) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    rt.set_sampler_params(model_id, 1.0, 1, 1.0);

    std::cout << "=== OpenCog RWKV Backend Test ===" << std::endl;
    std::cout << "Model loaded with cognitive enhancement layer." << std::endl << std::endl;

    // Test 1: basic generation
    std::cout << "[Test 1] Single-turn generation" << std::endl;
    std::vector<std::string> input_list = {
        "Hello!",
        "Hello! I am an AI assistant enhanced with OpenCog cognitive reasoning.",
        "What can you tell me about artificial intelligence?",
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    rt.chat(model_id, input_list, 50, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Response: " << rt.get_response_buffer_content(model_id) << std::endl;
    std::cout << "Generation time: " << elapsed << "s" << std::endl << std::endl;

    // Test 2: second turn demonstrating state caching
    std::cout << "[Test 2] Follow-up (tests state management)" << std::endl;
    input_list = {
        "Hello!",
        "Hello! I am an AI assistant enhanced with OpenCog cognitive reasoning.",
        "What can you tell me about artificial intelligence?",
        rt.get_response_buffer_content(model_id),
        "Can you elaborate on machine learning?",
    };

    rt.chat(model_id, input_list, 50, nullptr);
    std::cout << "Response: " << rt.get_response_buffer_content(model_id) << std::endl << std::endl;

    // Test 3: state serialization round-trip
    std::cout << "[Test 3] State serialization / deserialization" << std::endl;
    {
        std::any state;
        int ret = rt.get_backend(model_id)->get_state(state);
        if (ret == rwkvmobile::RWKV_SUCCESS && state.has_value()) {
            std::vector<uint8_t> serialized;
            ret = rt.get_backend(model_id)->serialize_runtime_state(state, serialized);
            if (ret == rwkvmobile::RWKV_SUCCESS) {
                std::any restored;
                ret = rt.get_backend(model_id)->deserialize_runtime_state(serialized, restored);
                if (ret == rwkvmobile::RWKV_SUCCESS && restored.has_value()) {
                    std::cout << "  State serialized (" << serialized.size()
                              << " bytes) and restored successfully." << std::endl;
                    rt.get_backend(model_id)->set_state(std::move(restored));
                } else {
                    std::cout << "  Deserialization failed (ret=" << ret << ")" << std::endl;
                }
            } else {
                std::cout << "  Serialization failed (ret=" << ret << ")" << std::endl;
            }
        } else {
            std::cout << "  get_state failed (ret=" << ret << ")" << std::endl;
        }
    }

    rt.release();
    std::cout << std::endl << "All tests completed." << std::endl;
    return 0;
}
