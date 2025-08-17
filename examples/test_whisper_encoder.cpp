#include <iostream>
#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <fstream>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "whisper.h"
#include "half.hpp"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void custom_sleep(int seconds) {
#if _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <tokenizer_file> <wav_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init_with_name(argv[5]);
    rwkvmobile_runtime_load_model(runtime, argv[1], argv[4], argv[3]);
    rwkvmobile_runtime_load_whisper_encoder(runtime, argv[2]);
    rwkvmobile_runtime_set_eos_token(runtime, "\x17");
    rwkvmobile_runtime_set_bos_token(runtime, "\x16");
    rwkvmobile_runtime_set_token_banned(runtime, {0}, 1);
    rwkvmobile_runtime_set_user_role(runtime, "");

    rwkvmobile_runtime_set_audio_prompt(runtime, argv[4]);

    rwkvmobile_runtime_eval_chat_async(runtime, "", 100, nullptr, 0);

    while (rwkvmobile_runtime_is_generating(runtime)) {
        custom_sleep(1);
    }

    struct response_buffer buffer = rwkvmobile_runtime_get_response_buffer_content(runtime);
    std::cout << buffer.content << std::endl;
    rwkvmobile_runtime_free_response_buffer(buffer);

    rwkvmobile_runtime_release_whisper_encoder(runtime);

    rwkvmobile_runtime_release(runtime);
    return 0;
}
