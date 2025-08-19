#ifndef VISION_ENCODER_H
#define VISION_ENCODER_H

#include "multimodal/multimodal_encoder.h"
#include "clip.h"
#include <memory>
#include <functional>

namespace rwkvmobile {

class VisionEncoder : public MultimodalEncoder {
public:
    VisionEncoder();
    ~VisionEncoder() override;

    int LoadModel(const std::string &model_path, const std::string &adapter_path) override;
    bool Encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens) override;

private:
    std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>> vision_encoder_ptr;
};

} // namespace rwkvmobile

#endif // VISION_ENCODER_H
