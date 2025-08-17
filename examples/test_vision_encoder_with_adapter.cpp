#include <iostream>
#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void custom_sleep(int seconds) {
#if _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

const char *msg0 = "Please analyze the picture and execute the following instruction or answer the following question: \\nplease recognize the data in the data in this picture";

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <adapter_file> <tokenizer_file> <image_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    rwkvmobile_runtime_load_model(runtime, argv[1], argv[6], argv[4]);
    rwkvmobile_runtime_load_vision_encoder_and_adapter(runtime, argv[2], argv[3]);
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});
    rwkvmobile_runtime_set_penalty_params(runtime, {0.0, 0.0, 0.0});
    rwkvmobile_runtime_set_eos_token(runtime, "\x17");
    rwkvmobile_runtime_set_bos_token(runtime, "\x16");
    rwkvmobile_runtime_set_token_banned(runtime, {0}, 1);

    rwkvmobile_runtime_set_image_prompt(runtime, argv[5]);
    const char *input_list[] = {msg0};

    rwkvmobile_runtime_eval_chat_with_history_async(runtime, input_list, 1, 500, nullptr, 0);

    while (rwkvmobile_runtime_is_generating(runtime)) {
        custom_sleep(1);
    }

    struct response_buffer buffer = rwkvmobile_runtime_get_response_buffer_content(runtime);
    std::cout << buffer.content << std::endl;
    rwkvmobile_runtime_free_response_buffer(buffer);

    rwkvmobile_runtime_release_vision_encoder(runtime);

    rwkvmobile_runtime_release(runtime);

    return 0;
}
