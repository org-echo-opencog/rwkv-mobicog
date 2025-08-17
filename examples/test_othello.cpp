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

void callback(const char *msg, const int, const char *next) {
    std::cout << msg;
}

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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init_with_name(argv[3]);
    rwkvmobile_runtime_load_model(runtime, argv[2], argv[3], argv[1]);
    rwkvmobile_runtime_set_penalty_params(runtime, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});

    std::string prompt = "<input>\n"
                        "● ● ● ● ● ● ● ● \n"
                        "● · ○ ○ ● ● ● ○ \n"
                        "● ○ ○ ○ ○ ● ● ○ \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● · · · · ● ● ● \n"
                        "● · · ○ ○ ○ ○ ○ \n"
                        "NEXT ● \n"
                        "MAX_WIDTH-2\n"
                        "MAX_DEPTH-2\n"
                        "</input>\n\n";

    std::cout << std::endl;

    rwkvmobile_runtime_gen_completion(runtime, prompt.c_str(), 64000, 0, callback);

    while (rwkvmobile_runtime_is_generating(runtime)) {
        custom_sleep(1);
    }

    return 0;
}
