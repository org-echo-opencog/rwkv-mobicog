#ifndef QNN_BACKEND_H
#define QNN_BACKEND_H

#include "backend.h"
#include "rwkv-qualcomm/Interfaces.hpp"
#include "rwkv-qualcomm/Utils/IOTensor.hpp"
#include "rmpack.h"

#ifndef _WIN32
#include <MNN/Interpreter.hpp>
#endif

namespace rwkvmobile {

class qnn_backend_context {
public:
    qnn_backend_context(std::string qnnBackendPath);

    ~qnn_backend_context();

    int qnn_create_power_config_id();
    int qnn_destory_power_config_id();
    int qnn_set_power_config();
    int qnn_register_op_package(std::string package_path, std::string interface_provider);
    int qnn_set_rpc_latency_and_polling();

    uint32_t powerConfigId = 0;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;

    std::string qnnBackendPath;
    std::string qnnBackendBasePath;
    void *qnnBackendLibraryHandle = nullptr;

    qnn::tools::rwkv_app::QnnFunctionPointers qnnFunctionPointers;

    Qnn_LogHandle_t qnnLogHandle = nullptr;
    Qnn_BackendHandle_t qnnBackendHandle = nullptr;
    Qnn_DeviceHandle_t qnnDeviceHandle = nullptr;

    IOTensor* qnnIOTensorUtils = nullptr;
};

class qnn_backend : public execution_provider {
public:
    ~qnn_backend() {
        if (state_head) {
            state_head->delete_after();
            state_head = nullptr;
        }
        release_model();
        release();
    }

    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, float *& logits) override;
    void free_logits_if_allocated(float *& logits) override {
        // persistant buffer, no need to free after use
        return;
    }

    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;
    double get_prefill_speed() override {
        return prefill_speed;
    }

    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;

    int copy_float_to_qnn_tensor(Qnn_Tensor_t *qnn_tensor, const float *buffer, size_t element_count);

    int copy_qnn_tensor_to_float(Qnn_Tensor_t *qnn_tensor, float *buffer, size_t element_count);

    int post_graph_execute(float *& logits);
private:
    double prefill_speed = -1;
    void *qnnModelHandle = nullptr;

    bool isContextCreated = false;
    bool isTensorInitialized = false;

    int prefillSequenceLength = 0;
    int embdPrefillSequenceLength = 0;

    std::vector<Qnn_ContextHandle_t> qnnContextHandles;

    uint32_t qnnDecodeGraphsCount = 0;
    GraphInfo_t **qnnDecodeGraphsInfo = nullptr;

    uint32_t qnnPrefillGraphsCount = 0;
    GraphInfo_t **qnnPrefillGraphsInfo = nullptr;

    uint32_t qnnEmbdGraphsCount = 0;
    GraphInfo_t **qnnEmbdGraphsInfo = nullptr;

    uint32_t qnnEmbdPrefillGraphsCount = 0;
    GraphInfo_t **qnnEmbdPrefillGraphsInfo = nullptr;

    uint32_t graphConfigsInfoCount = 0;
    GraphConfigInfo_t **graphConfigsInfo = nullptr;

    Qnn_Tensor_t *inputTensors[8] = {nullptr};
    Qnn_Tensor_t *outputTensors[8] = {nullptr};

    Qnn_Tensor_t *inputTensorsPrefill[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsPrefill[8] = {nullptr};

    Qnn_Tensor_t *inputTensorsEmbd[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsEmbd[8] = {nullptr};

    Qnn_Tensor_t *inputTensorsEmbdPrefill[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsEmbdPrefill[8] = {nullptr};

    Qnn_Tensor_t *logitsOutputTensor = nullptr;
    Qnn_Tensor_t *tokenInputTensor = nullptr;
    Qnn_Tensor_t *tokenInputTensorPrefill = nullptr;
    Qnn_Tensor_t *tokenInputTensorEmbd = nullptr;
    Qnn_Tensor_t *tokenInputTensorEmbdPrefill = nullptr;

    size_t logitsOutputTensorSize = 0;

    std::vector<std::unordered_map<std::string, void*>> decodeGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> decodeGraphsTensorNameToSize;
    std::vector<std::unordered_map<std::string, void*>> prefillGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> prefillGraphsTensorNameToSize;
    std::vector<std::unordered_map<std::string, void*>> embdGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> embdGraphsTensorNameToSize;
    std::vector<std::unordered_map<std::string, void*>> embdPrefillGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> embdPrefillGraphsTensorNameToSize;

    std::unordered_map<std::string, void*> stateTensorsNameToTensorPointer;

    int qnn_initialize_tensors();

    void fill_quantized_tensor(float value, Qnn_Tensor_t *tensor);

    int execute_graph(GraphInfo_t** graphInfo, int graphsCount, Qnn_Tensor_t** inputTensors, Qnn_Tensor_t** outputTensors);
    int execute_decode_graph();
    int execute_prefill_graph();
    int execute_emb_decode_graph();
    int execute_emb_prefill_graph();

    std::vector<float> logits_buffer;

    std::shared_ptr<uint8_t> external_embeddings = nullptr;
    std::string external_lmhead_filetype = "None";
#ifndef _WIN32
    MNN::Interpreter *external_lmhead_interpretor = nullptr;
    MNN::Session *external_lmhead_mnn_session = nullptr;
    MNN::Tensor *external_lmhead_input_tensor = nullptr;
#endif

#ifndef _WIN32
    RMPack *rmpack = nullptr;
#endif
};

}

#endif
