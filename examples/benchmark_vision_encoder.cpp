#include <iostream>
#include <chrono>
#include "../src/multimodal/vision/clip.h"
#include "../src/multimodal/vision/llava.h"

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model> <adapter> <image>" << std::endl;
        return 1;
    }

    clip_ctx * ctx = clip_model_load(argv[1], argv[2], 0);
    if (ctx == NULL) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    double total_time = 0;
    int n_runs = 10;
    for (int i = 0; i < n_runs; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        auto embd = llava_image_embed_make_with_filename(ctx, 4, argv[3]);
        if (embd == NULL) {
            std::cerr << "Failed to embed image" << std::endl;
            return 1;
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        llava_image_embed_free(embd);
    }
    std::cout << "Average time: " << total_time / n_runs << " ms" << std::endl;

    clip_free(ctx);
    return 0;
}
