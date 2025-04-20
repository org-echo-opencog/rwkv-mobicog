#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "cosyvoice.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <encoder_path> <tokenizer_file> <wav_file>" << std::endl;
        return 1;
    }
    rwkvmobile::runtime runtime;
    runtime.init(argv[2]);
    runtime.load_model(argv[1]);

    std::string encoder_path = argv[3];

    runtime.cosyvoice_load_models(
        encoder_path + "speech_tokenizer_v2.onnx",
        encoder_path + "campplus.onnx",
        encoder_path + "flow_encoder.fp16.onnx",
        encoder_path + "flow_decoder_estimator.ncnn.bin",
        encoder_path + "hift.onnx",
        encoder_path + std::string(argv[4]),
        encoder_path + "spk_info.msgpack"
    );

    std::string tts_text = "要陪我去拍照吗？";
    std::string instruction_text = "";
    // std::string instruction_text = ""; // empty string means no instruction

    // std::cout << runtime.cosyvoice_get_spk_names() << std::endl;

    runtime.run_tts(tts_text, instruction_text, argv[5], "test.wav");
    runtime.run_tts_with_predefined_spks(tts_text, instruction_text, "March 7th_Chinese(PRC)", "test2.wav");

    // runtime.run_tts_with_predefined_spks(tts_text, instruction_text, "Blade_Japanese", "test2.wav");

    runtime.release();

    return 0;
}
