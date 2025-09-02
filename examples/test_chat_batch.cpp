#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "logger.h"

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

    std::vector<std::string> input_list = {
        "Hello!",
        "Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.",
        "随机说一个两位数",
    };
    std::cout << "Testing batch chat prompt: " << input_list[input_list.size()-1] << std::endl << std::endl;
    runtime.chat_batch(model_id, input_list, 300, 4, nullptr, false);
    auto batch_response = runtime.get_response_buffer_content_batch(model_id);
    for (int i = 0; i < 4; i++) {
        std::cout << "Response " << i << ": " << batch_response[i] << std::endl << std::endl;
    }

    input_list.push_back(batch_response[3]);
    input_list.push_back("复述一遍你刚才说的两位数");

    std::cout << "Testing new chat prompt: " << input_list[input_list.size()-1] << std::endl << std::endl;
    runtime.chat(model_id, input_list, 300, nullptr, false);
    std::cout << "Response: " << runtime.get_response_buffer_content(model_id) << std::endl;

    runtime.release();

    return 0;
}
