#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <state_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime runtime;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.init(argv[4]), "Failed to initialize runtime");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_tokenizer(argv[1]), "Failed to load tokenizer");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_model(argv[3]), "Failed to load model");
    // runtime.set_sampler_params(1.0, 1, 1.0);

    runtime.load_initial_state(argv[2]);

    runtime.set_prompt("System: 请你扮演名为白素贞的角色，你的设定是：你来自《新白娘子传奇》，你叫白素贞。你是一条修炼千年的蛇仙，为了报答救命之恩，你化为人形来到人间，与许仙相识相爱。然而，因误会而分离的故事情节使得两人的感情曲折离奇，最终有情人终成眷属。\n\n");

    std::vector<std::string> input_list = {
        "娘子我来了",
    };
    runtime.chat(input_list, 1000, nullptr);
    std::cout << "Response: " << runtime.get_response_buffer_content() << std::endl;

    std::cout << std::endl;

    runtime.release();

    return 0;
}
