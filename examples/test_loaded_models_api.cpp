#include <iostream>
#include <chrono>
#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "commondef.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void test_get_loaded_models_info(rwkvmobile_runtime_t runtime) {
    std::cout << "\n=== Testing Get Loaded Models Info ===" << std::endl;

    // Test getting loaded model IDs list
    int model_ids[10];
    int count = rwkvmobile_runtime_get_loaded_model_ids(runtime, model_ids, 10);

    std::cout << "Number of loaded models: " << count << std::endl;
    std::cout << "Model ID list: ";
    for (int i = 0; i < count; i++) {
        std::cout << model_ids[i];
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Test getting detailed model info
    struct loaded_models_list models_list = rwkvmobile_runtime_get_loaded_models_info(runtime);

    std::cout << "\nDetailed model information:" << std::endl;
    for (int i = 0; i < models_list.count; i++) {
        struct model_info* model = &models_list.models[i];

        std::cout << "Model " << (i + 1) << ":" << std::endl;
        std::cout << "  Model ID: " << model->model_id << std::endl;
        std::cout << "  Model Path: " << (model->model_path ? model->model_path : "N/A") << std::endl;
        std::cout << "  Backend Name: " << (model->backend_name ? model->backend_name : "N/A") << std::endl;
        std::cout << "  Tokenizer Path: " << (model->tokenizer_path ? model->tokenizer_path : "N/A") << std::endl;
        std::cout << "  User Role: " << (model->user_role ? model->user_role : "N/A") << std::endl;
        std::cout << "  Response Role: " << (model->response_role ? model->response_role : "N/A") << std::endl;
        std::cout << "  BOS Token: " << (model->bos_token ? model->bos_token : "N/A") << std::endl;
        std::cout << "  EOS Token: " << (model->eos_token ? model->eos_token : "N/A") << std::endl;
        std::cout << "  Thinking Token: " << (model->thinking_token ? model->thinking_token : "N/A") << std::endl;
        std::cout << "  Is Generating: " << (model->is_generating ? "Yes" : "No") << std::endl;
        std::cout << "  Vocab Size: " << model->vocab_size << std::endl;
        std::cout << std::endl;
    }

    // Free memory
    rwkvmobile_runtime_free_loaded_models_list(models_list);
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> [second_model_file]" << std::endl;
        return 1;
    }

    std::cout << "=== RWKV Mobile Loaded Models API Test ===" << std::endl;

    // Create runtime
    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    if (runtime == nullptr) {
        std::cerr << "Failed to initialize runtime" << std::endl;
        return 1;
    }

    std::cout << "Runtime initialized successfully" << std::endl;

    // Test before loading model (should return empty list)
    std::cout << "\n--- Testing Before Model Load ---" << std::endl;
    test_get_loaded_models_info(runtime);

    // Load first model
    std::cout << "Loading model..." << std::endl;
    int ret = rwkvmobile_runtime_load_model(runtime, argv[2], argv[3], argv[1]);
    ENSURE_SUCCESS_OR_LOG_EXIT(ret, "Failed to load model");

    std::cout << "Model loaded successfully" << std::endl;

    // Test getting loaded models list
    test_get_loaded_models_info(runtime);

    // Try loading second model if available
    // This is just an example, modify model path as needed in actual use
    if (argc > 4) {
        std::cout << "Loading second model..." << std::endl;
        int model_id = rwkvmobile_runtime_load_model(runtime, argv[4], argv[3], argv[1]);
        ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
        std::cout << "Second model loaded successfully" << std::endl;
        test_get_loaded_models_info(runtime);
    }

    // Release runtime
    rwkvmobile_runtime_release(runtime);
    std::cout << "\nTest completed!" << std::endl;

    return 0;
}
