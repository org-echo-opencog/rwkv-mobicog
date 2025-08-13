#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <any>

#ifdef ENABLE_QNN
#include "qnn_backend.h"
#include "commondef.h"
#include "soc_detect.h"
#include "PAL/DynamicLoading.hpp"
#include "DynamicLoadUtil.hpp"
#include "DataUtil.hpp"
#include "Utils.hpp"
#include "QnnTypeMacros.hpp"
#include "rmpack.h"
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>
#include <QnnContext.h>
#endif

#include "logger.h"
#include "half.hpp"

#ifdef _WIN32
#define USE_MMAP 0
#else
#define USE_MMAP 1
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define ENABLE_QNN 1

#define DEFAULT_QNN_LOGLEVEL QNN_LOG_LEVEL_ERROR

namespace rwkvmobile {

#ifdef ENABLE_QNN
using namespace qnn::tools;

static void logCallback(const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp) {
    char buffer[1024];

    // TODO
    try {
        vsnprintf(buffer, sizeof(buffer), fmt, argp);
        LOGI("[QNN] %s", buffer);
    } catch (const std::exception& e) {
    }

    return;
}

static void getTensorDims(std::vector<size_t>& dims,
    uint32_t* inDimensions,
    uint32_t rank) {
    if (nullptr == inDimensions) {
        LOGE("input dimensions is nullptr");
        return;
    }
    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
}

static size_t getQnnDatatypeSize(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16:
            return sizeof(uint16_t);
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
            return sizeof(uint32_t);
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_BOOL_8:
            return sizeof(uint8_t);
        case QNN_DATATYPE_FLOAT_64:
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return sizeof(uint64_t);
        default:
            LOGE("Unsupported data type");
            return 0;
    }
}

int qnn_backend::init(void * extra) {
    if (extra != nullptr) {
        qnnBackendPath = std::string((char *)extra);
        LOGI("Setting QNN Backend Path: %s\n", qnnBackendPath.c_str());
    } else {
        qnnBackendPath = "libQnnHtp.so";
        LOGI("Using default QNN Backend Path: %s\n", qnnBackendPath.c_str());
    }

    return RWKV_SUCCESS;
}

int qnn_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    if (qnnBackendPath.empty()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    bool is_context_cache = 
#ifdef WIN32
        model_path.find(".dll") == std::string::npos;
#else
        model_path.find(".so") == std::string::npos;
#endif

    bool is_rmpack = model_path.find(".rmpack") != std::string::npos;
    if (is_rmpack) {
#ifndef _WIN32
        try {
            rmpack = new RMPack(model_path);
        } catch (const std::exception& e) {
            LOGE("Error loading rmpack: %s", e.what());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        }
#else
        LOGE("TODO: add windows support for rmpack");
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
#endif
    }

    // load QNN functions
    auto qnnStatusCode = dynamicloadutil::getQnnFunctionPointers(
        qnnBackendPath, model_path, &qnnFunctionPointers, &qnnBackendLibraryHandle, !is_context_cache, &qnnModelHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != qnnStatusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == qnnStatusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load backend: %s", qnnBackendPath.c_str());
            return RWKV_ERROR_BACKEND | RWKV_ERROR_IO;
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == qnnStatusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load model:%s ", model_path.c_str());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        } else {
            LOGE("Error initializing QNN Function Pointers");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    if (is_context_cache) {
        std::string qnnSystemLibPath = 
#ifdef WIN32
            qnnBackendPath.substr(0, qnnBackendPath.find("QnnHtp.dll")) + "QnnSystem.dll";
#else
            qnnBackendPath.substr(0, qnnBackendPath.find("libQnnHtp.so")) + "libQnnSystem.so";
#endif

        auto qnnSystemLibStatus = dynamicloadutil::getQnnSystemFunctionPointers(qnnSystemLibPath, &qnnFunctionPointers);
        if (dynamicloadutil::StatusCode::SUCCESS != qnnSystemLibStatus) {
            LOGE("Error initializing QNN System Function Pointers");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    bool usingHtp = qnnBackendPath.find("Htp") != std::string::npos;
    if (usingHtp) {
        LOGI("Using QNN HTP Backend");
    }
    else {
        LOGE("Do not use QNN CPU/GPU backends!");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    // initialize QNN logging
    auto logLevel = DEFAULT_QNN_LOGLEVEL;
    if (QNN_SUCCESS !=
        qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &qnnLogHandle)) {
      LOGW("Unable to initialize logging in the backend.");
    }

    // initialize QNN backend
    auto qnnBackendStatus = qnnFunctionPointers.qnnInterface.backendCreate(
        qnnLogHandle, nullptr, &qnnBackendHandle);
    if (QNN_BACKEND_NO_ERROR != qnnBackendStatus) {
      LOGE("Could not initialize backend due to error = %lu", qnnBackendStatus);
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    LOGI("Initialize Backend Returned Status = %lu", qnnBackendStatus);

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnDevicePropertyStatus) {
            LOGE("Device property is not known to backend");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }

        auto qnnCreateDeviceStatus = qnnFunctionPointers.qnnInterface.deviceCreate(qnnLogHandle, nullptr, &qnnDeviceHandle);
        if (QNN_SUCCESS != qnnCreateDeviceStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnCreateDeviceStatus) {
            LOGE("Failed to create device");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    qnnIOTensorUtils = new IOTensor(BufferAlloc::SHARED_BUFFER, &qnnFunctionPointers.qnnInterface);

    if (usingHtp) {
        if (RWKV_SUCCESS != qnn_create_power_config_id()) {
            LOGE("Could not create HTP power config id");
        } else {
            if (RWKV_SUCCESS != qnn_set_rpc_latency_and_polling()) {
                LOGE("Could not set HTP rpc latency and polling");
            } else if (RWKV_SUCCESS != qnn_set_power_config()) {
                LOGE("Could not set HTP power config");
            }
        }

        soc_detect soc_detect;
        soc_detect.detect_platform();
        std::string htp_arch = soc_detect.get_htp_arch();
        std::string custom_op_name = "libQnnRwkvWkvOpPackage.so";
        if (htp_arch == "v79") {
            custom_op_name = "libQnnRwkvWkvOpPackageV79.so";
        } else if (htp_arch == "v75") {
            custom_op_name = "libQnnRwkvWkvOpPackageV75.so";
        } else if (htp_arch == "v73") {
            custom_op_name = "libQnnRwkvWkvOpPackageV73.so";
        } else if (htp_arch == "v69") {
            custom_op_name = "libQnnRwkvWkvOpPackageV69.so";
        } else if (htp_arch == "v68") {
            custom_op_name = "libQnnRwkvWkvOpPackageV68.so";
        }

        std::vector<std::string> paths;
        if (!extra_str.empty()) {
            paths.push_back(extra_str);
        } else {
#ifndef _WIN32
            const char* ldLibraryPath = getenv("LD_LIBRARY_PATH");
            if (ldLibraryPath) {
                std::string pathStr(ldLibraryPath);
                std::stringstream ss(pathStr);
                std::string dir;
                while (std::getline(ss, dir, ':')) {
                    paths.push_back(dir);
                }
            }
#endif
        }

        for (auto dir : paths) {
            std::string fullPath = dir + "/" + custom_op_name;
            std::ifstream file(fullPath);
            if (file.good()) {
                LOGI("Found %s in path: %s", custom_op_name.c_str(), fullPath.c_str());
                if (RWKV_SUCCESS != qnn_register_op_package(fullPath, "RwkvWkvOpPackageInterfaceProvider")) {
                    LOGE("Op package registration failed");
                }
                break;
            }
        }
    }

    if (is_context_cache || is_rmpack) {
        if (nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
            nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
            nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
            LOGE("QNN System function pointers are not populated.");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }

        Qnn_ContextHandle_t first_contextHandle{nullptr};
        QnnHtpContext_CustomConfig_t customConfigSF;
        customConfigSF.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;

        std::vector<std::shared_ptr<uint8_t>> buffer;
        std::vector<uint64_t> bufferSizes;

        int n_chunks = 1;
        size_t pos = 0;
        int spill_fill_buffer_size = 0;
        if (is_rmpack) {
            n_chunks = rmpack->getConfig()["n_chunks"];
            spill_fill_buffer_size = rmpack->getConfig()["spill_fill_buffer_size"];
        } else {
            pos = model_path.find("_chunk");
            if (pos != std::string::npos) {
                n_chunks = std::stoi(model_path.substr(model_path.find("of") + 2));
                LOGI("Number of chunks: %d", n_chunks);
                if (n_chunks == 4) {
                    spill_fill_buffer_size = 320000000;
                }
            }
        }

        buffer.resize(n_chunks);
        bufferSizes.resize(n_chunks);
        qnnContextHandles.resize(n_chunks);

        int returnStatus = RWKV_SUCCESS;
        std::vector<GraphInfo_t **> graphInfos(n_chunks);
        std::vector<uint32_t> graphCounts(n_chunks);

        // read model binaries
        datautil::StatusCode binaryReadingStatus{datautil::StatusCode::SUCCESS};
        for (int i = 0; i < n_chunks; i++) {
            // get file size and read file to memory / mmap file
            if (is_rmpack) {
                bufferSizes[i] = rmpack->getFileSize("model_" + std::to_string(i));
#if USE_MMAP
                buffer[i] = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->mmapFile("model_" + std::to_string(i)), [this, i](uint8_t* p) {
                        if (p) {
                            rmpack->unmapFile("model_" + std::to_string(i));
                        }
                    }
                );
#else
                buffer[i] = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->readFileToMemory("model_" + std::to_string(i)), [this, i](uint8_t* p) {
                        if (p) {
                            rmpack->freeMemory("model_" + std::to_string(i));
                        }
                    }
                );
#endif
            } else {
                if (n_chunks > 1) {
                    model_path = model_path.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
                    std::cout << "Reading chunk: " << model_path << std::endl;
                }
                std::tie(binaryReadingStatus, bufferSizes[i]) = datautil::getFileSize(model_path);
                if (0 == bufferSizes[i]) {
                    LOGE("Received path to an empty file. Nothing to deserialize.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
                std::cout << "Buffer size: " << bufferSizes[i] << std::endl;

#if USE_MMAP
                int fd = open(model_path.c_str(), O_RDONLY);
                if (fd < 0) {
                    LOGE("Failed to open file %s", model_path.c_str());
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }

                buffer[i] = std::shared_ptr<uint8_t>(
                    (uint8_t*)mmap(NULL, bufferSizes[i], PROT_READ, MAP_SHARED, fd, 0), [bufferSizes, i](uint8_t* p) {
                        if (p) {
                            munmap(p, bufferSizes[i]);
                        }
                    }
                );

                if (buffer[i].get() == MAP_FAILED) {
                    LOGE("Failed to mmap file %s", model_path.c_str());
                    close(fd);
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#else
                buffer[i] = std::shared_ptr<uint8_t>(new uint8_t[bufferSizes[i]], std::default_delete<uint8_t[]>());
                if (!buffer[i]) {
                    LOGE("Failed to allocate memory.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC;
                }

                binaryReadingStatus = datautil::readBinaryFromFile(
                    model_path, reinterpret_cast<uint8_t*>(buffer[i].get()), bufferSizes[i]);
                if (binaryReadingStatus != datautil::StatusCode::SUCCESS) {
                    LOGE("Failed to read binary data.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#endif
            }
            // inspect binary info
            QnnSystemContext_Handle_t sysCtxHandle{nullptr};
            if (QNN_SUCCESS != qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
                LOGE("Could not create system handle.");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
            Qnn_ContextBinarySize_t binaryInfoSize{0};
            if (RWKV_SUCCESS == returnStatus &&
                QNN_SUCCESS != qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                                    sysCtxHandle,
                                    static_cast<void*>(buffer[i].get()),
                                    bufferSizes[i],
                                    &binaryInfo,
                                    &binaryInfoSize)) {
                LOGE("Failed to get context binary info");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            // fill GraphInfo_t based on binary info
            if (RWKV_SUCCESS == returnStatus &&
                !qnn::tools::rwkv_app::copyMetadataToGraphsInfo(binaryInfo, graphInfos[i], graphCounts[i])) {
                LOGE("Failed to copy metadata.");
                returnStatus = RWKV_ERROR_MODEL;
            }
            qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
            sysCtxHandle = nullptr;

            if (RWKV_SUCCESS == returnStatus &&
                nullptr == qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
                LOGE("contextCreateFromBinaryFnHandle is nullptr.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            // make custom configs
            QnnHtpContext_CustomConfig_t ioMemEstimation;
            ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
            ioMemEstimation.ioMemEstimation = true;

            QnnContext_Config_t** cfgs{nullptr};

            int cfgs_count = 1;
            if (spill_fill_buffer_size > 0) {
                cfgs_count++;
            }

            cfgs                  = (QnnContext_Config_t**)malloc((cfgs_count + 1) * sizeof(QnnContext_Config_t*));
            cfgs[0]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
            cfgs[0]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            cfgs[0]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);

            if (spill_fill_buffer_size > 0) {
                QnnHtpContext_GroupRegistration_t groupInfo{nullptr};
                if (i == 0) {
                    groupInfo.firstGroupHandle = 0x0;
                } else {
                    groupInfo.firstGroupHandle = first_contextHandle;
                }
                groupInfo.maxSpillFillBuffer = spill_fill_buffer_size;
                customConfigSF.groupRegistration = groupInfo;

                cfgs[cfgs_count - 1]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
                cfgs[cfgs_count - 1]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
                cfgs[cfgs_count - 1]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&customConfigSF);
            }

            cfgs[cfgs_count] = nullptr;


            if (RWKV_SUCCESS == returnStatus &&
                qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                    qnnBackendHandle,
                    qnnDeviceHandle,
                    (const QnnContext_Config_t**)cfgs,
                    static_cast<void*>(buffer[i].get()),
                    bufferSizes[i],
                    &qnnContextHandles[i],
                    nullptr)) {
                LOGE("Could not create context from binary.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            for (int j = 0; j < cfgs_count; j++) {
                free(cfgs[j]);
                cfgs[j] = nullptr;
            }
            free(cfgs);
            cfgs = nullptr;

            isContextCreated = true;
            if (RWKV_SUCCESS == returnStatus) {
                for (size_t graphIdx = 0; graphIdx < graphCounts[i]; graphIdx++) {
                    if (nullptr == qnnFunctionPointers.qnnInterface.graphRetrieve) {
                        LOGE("graphRetrieveFnHandle is nullptr.");
                        returnStatus = RWKV_ERROR_MODEL;
                        break;
                    }
                    if (QNN_SUCCESS !=
                        qnnFunctionPointers.qnnInterface.graphRetrieve(
                            qnnContextHandles[i], (*graphInfos[i])[graphIdx].graphName, &((*graphInfos[i])[graphIdx].graph))) {
                        LOGE("Unable to retrieve graph handle for graph Idx: %zu", graphIdx);
                        returnStatus = RWKV_ERROR_MODEL;
                    }
                }
            }
            if (RWKV_SUCCESS != returnStatus) {
                LOGD("Cleaning up graph Info structures.");
                freeGraphsInfo(&graphInfos[i], graphCounts[i]);
            }

            if (RWKV_SUCCESS == returnStatus && i == 0) {
                first_contextHandle = qnnContextHandles[i];
            }
        }

        buffer.clear();

        qnnDecodeGraphsCount = 0;
        qnnPrefillGraphsCount = 0;
        qnnEmbdGraphsCount = 0;
        qnnEmbdPrefillGraphsCount = 0;
        for (int i = 0; i < n_chunks; i++) {
            for (int j = 0; j < graphCounts[i]; j++) {
                auto graphName = std::string((*graphInfos[i])[j].graphName);
                if (graphName.find("ext_embedding_prefill") != std::string::npos) {
                    qnnEmbdPrefillGraphsCount++;
                } else if (graphName.find("ext_embedding") != std::string::npos) {
                    qnnEmbdGraphsCount++;
                } else if (graphName.find("prefill") != std::string::npos) {
                    qnnPrefillGraphsCount++;
                } else {
                    qnnDecodeGraphsCount++;
                }
            }
        }

        qnnDecodeGraphsInfo = (GraphInfo_t **)calloc(qnnDecodeGraphsCount, sizeof(GraphInfo_t *));
        GraphInfo_t *graphInfoArrDecode =
            (GraphInfo_t *)calloc(qnnDecodeGraphsCount, sizeof(GraphInfo_t));

        GraphInfo_t *graphInfoArrPrefill = nullptr;
        GraphInfo_t *graphInfoArrEmbd = nullptr;
        GraphInfo_t *graphInfoArrEmbdPrefill = nullptr;

        if (qnnPrefillGraphsCount > 0) {
            qnnPrefillGraphsInfo = (GraphInfo_t **)calloc(qnnPrefillGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrPrefill =
                (GraphInfo_t *)calloc(qnnPrefillGraphsCount, sizeof(GraphInfo_t));
        }
        if (qnnEmbdGraphsCount > 0) {
            qnnEmbdGraphsInfo = (GraphInfo_t **)calloc(qnnEmbdGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrEmbd =
                (GraphInfo_t *)calloc(qnnEmbdGraphsCount, sizeof(GraphInfo_t));
        }
        if (qnnEmbdPrefillGraphsCount > 0) {
            qnnEmbdPrefillGraphsInfo = (GraphInfo_t **)calloc(qnnEmbdPrefillGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrEmbdPrefill =
                (GraphInfo_t *)calloc(qnnEmbdPrefillGraphsCount, sizeof(GraphInfo_t));
        }

        if (nullptr == qnnDecodeGraphsInfo || nullptr == graphInfoArrDecode ||
            (qnnPrefillGraphsCount > 0 && (nullptr == qnnPrefillGraphsInfo || nullptr == graphInfoArrPrefill)) ||
            (qnnEmbdGraphsCount > 0 && (nullptr == qnnEmbdGraphsInfo || nullptr == graphInfoArrEmbd)) ||
            (qnnEmbdPrefillGraphsCount > 0 && (nullptr == qnnEmbdPrefillGraphsInfo || nullptr == graphInfoArrEmbdPrefill))) {
            LOGE("Failed to allocate memory for *graphInfo");
            if (nullptr != qnnDecodeGraphsInfo) {
                free(qnnDecodeGraphsInfo);
            }
            if (nullptr != qnnPrefillGraphsInfo) {
                free(qnnPrefillGraphsInfo);
            }
            if (nullptr != qnnEmbdGraphsInfo) {
                free(qnnEmbdGraphsInfo);
            }
            if (nullptr != qnnEmbdPrefillGraphsInfo) {
                free(qnnEmbdPrefillGraphsInfo);
            }
            if (nullptr != graphInfoArrDecode) {
                free(graphInfoArrDecode);
            }
            if (nullptr != graphInfoArrPrefill) {
                free(graphInfoArrPrefill);
            }
            if (nullptr != graphInfoArrEmbd) {
                free(graphInfoArrEmbd);
            }
            if (nullptr != graphInfoArrEmbdPrefill) {
                free(graphInfoArrEmbdPrefill);
            }
            returnStatus = RWKV_ERROR_MODEL;
        }
        LOGI("qnnDecodeGraphsCount: %d, qnnPrefillGraphsCount: %d, qnnEmbdGraphsCount: %d, qnnEmbdPrefillGraphsCount: %d", qnnDecodeGraphsCount, qnnPrefillGraphsCount, qnnEmbdGraphsCount, qnnEmbdPrefillGraphsCount);

        if (RWKV_SUCCESS == returnStatus) {
            int prefill_gidx = 0, decode_gidx = 0, embd_gidx = 0, embd_prefill_gidx = 0;
            for (int i = 0; i < n_chunks; i++) {
                for (int j = 0; j < graphCounts[i]; j++) {
                    auto graphName = std::string((*graphInfos[i])[j].graphName);
                    if (graphName.find("ext_embedding_prefill") != std::string::npos) {
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx] = graphInfoArrEmbdPrefill + embd_prefill_gidx;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        embd_prefill_gidx++;
                    } else if (graphName.find("ext_embedding") != std::string::npos) {
                        qnnEmbdGraphsInfo[embd_gidx] = graphInfoArrEmbd + embd_gidx;
                        qnnEmbdGraphsInfo[embd_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnEmbdGraphsInfo[embd_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnEmbdGraphsInfo[embd_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        embd_gidx++;
                    } else if (graphName.find("prefill") != std::string::npos) {
                        qnnPrefillGraphsInfo[prefill_gidx] = graphInfoArrPrefill + prefill_gidx;
                        qnnPrefillGraphsInfo[prefill_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnPrefillGraphsInfo[prefill_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnPrefillGraphsInfo[prefill_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        prefill_gidx++;
                    } else {
                        qnnDecodeGraphsInfo[decode_gidx] = graphInfoArrDecode + decode_gidx;
                        qnnDecodeGraphsInfo[decode_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnDecodeGraphsInfo[decode_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnDecodeGraphsInfo[decode_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnDecodeGraphsInfo[decode_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnDecodeGraphsInfo[decode_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnDecodeGraphsInfo[decode_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        decode_gidx++;
                    }
                }
            }
        }
    } else {
        // create context
        qnnContextHandles.resize(1);
        if (QNN_CONTEXT_NO_ERROR != qnnFunctionPointers.qnnInterface.contextCreate(
                    qnnBackendHandle,
                    qnnDeviceHandle,
                    nullptr, // const QnnContext_Config_t**
                    &qnnContextHandles[0])) {
            LOGE("Could not create context");
            return RWKV_ERROR_BACKEND;
        }
        isContextCreated = true;

        // conpose graphs
        if (graphConfigsInfo == nullptr) {
            graphConfigsInfoCount = 2;

            graphConfigsInfo = new GraphConfigInfo_t*[graphConfigsInfoCount];
            graphConfigsInfo[0] = new GraphConfigInfo_t();
            graphConfigsInfo[0]->graphName = (char*)"model";
            graphConfigsInfo[0]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
            graphConfigsInfo[1] = new GraphConfigInfo_t();
            graphConfigsInfo[1]->graphName = (char*)"model_fp16";
            graphConfigsInfo[1]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
        
            static QnnHtpGraph_CustomConfig_t customConfig;
            customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
            customConfig.precision = QNN_PRECISION_FLOAT16;
            static QnnGraph_Config_t graphConfig;
            graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graphConfig.customConfig = &customConfig;
            for (int i = 0; i < graphConfigsInfoCount; i++) {
                graphConfigsInfo[i]->graphConfigs[0] = &graphConfig;
                graphConfigsInfo[i]->graphConfigs[1] = nullptr;
            }
        }

        if (ModelError_t::MODEL_NO_ERROR !=
            qnnFunctionPointers.composeGraphsFnHandle(
                qnnBackendHandle,
                qnnFunctionPointers.qnnInterface,
                qnnContextHandles[0],
                (const GraphConfigInfo_t**)graphConfigsInfo,
                graphConfigsInfoCount,
                &qnnDecodeGraphsInfo,
                &qnnDecodeGraphsCount,
                false,
                logCallback,
                DEFAULT_QNN_LOGLEVEL)) {
          LOGE("Failed in composeGraphs()");
          return RWKV_ERROR_MODEL;
        }

        // finalize graphs
        for (size_t graphIdx = 0; graphIdx < qnnDecodeGraphsCount; graphIdx++) {
            if (QNN_GRAPH_NO_ERROR !=
                qnnFunctionPointers.qnnInterface.graphFinalize(
                    (*qnnDecodeGraphsInfo)[graphIdx].graph, nullptr, nullptr)) {
                return RWKV_ERROR_MODEL;
            }
        }

        // save context cache
// #if WIN32
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.dll')) + "_cache.bin");
// #else
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.so')) + "_cache.bin");
// #endif

    }

    if (RWKV_SUCCESS != qnn_initialize_tensors()) {
        LOGE("Could not initialize tensors");
        return RWKV_ERROR_MODEL;
    }

    int state_tensor_count = 0;
    for (int i = 0; i < decodeGraphsTensorNameToTensorPointer.size(); i++) {
        for (auto &[tensorName, tensor] : decodeGraphsTensorNameToTensorPointer[i]) {
            if (tensorName.find("state") != std::string::npos && tensorName.find("in") != std::string::npos) {
                state_tensor_count++;
            }
        }
    }
    n_layers = state_tensor_count / 3;

    std::vector<size_t> dims_state;
    getTensorDims(dims_state, QNN_TENSOR_GET_DIMENSIONS((Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[0]["state1_in"]), QNN_TENSOR_GET_RANK((Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[0]["state1_in"]));
    num_heads = dims_state[0];
    hidden_size = num_heads * dims_state[1];

    if (rmpack != nullptr) {
        vocab_size = rmpack->getConfig()["vocab_size"];
    } else {
        std::vector<size_t> dims;
        getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(logitsOutputTensor), QNN_TENSOR_GET_RANK(logitsOutputTensor));
        vocab_size = dims[2];
    }

    if (rmpack != nullptr) {
        int use_external_embedding = rmpack->getConfig()["use_external_embedding"];
        if (use_external_embedding) {
            // let's assert that the embedding dtype is the same as the input tensor dtype
            try {
#if USE_MMAP
                external_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->mmapFile("embedding"),
                    [this](uint8_t* p) {
                        rmpack->unmapFile("embedding");
                    }
                );
#else
                external_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->readFileToMemory("embedding"),
                    [this](uint8_t* p) {
                        rmpack->freeMemory("embedding");
                    }
                );
#endif
            } catch (const std::exception& e) {
                LOGE("Failed to load external embedding: %s", e.what());
                return RWKV_ERROR_MODEL;
            }
        }

        int use_external_lmhead = rmpack->getConfig()["use_external_lmhead"];
        if (use_external_lmhead) {
            external_lmhead_filetype = rmpack->getConfig()["external_lmhead_filetype"];
            if (external_lmhead_filetype == "mnn") {
#ifndef _WIN32
                try {
#if USE_MMAP
                    void* buffer = rmpack->mmapFile("lmhead");
#else
                    void* buffer = rmpack->readFileToMemory("lmhead");
#endif
                    external_lmhead_interpretor = MNN::Interpreter::createFromBuffer(buffer, rmpack->getFileInfo("lmhead")->size);
                    MNN::ScheduleConfig conf;
                    external_lmhead_mnn_session = external_lmhead_interpretor->createSession(conf);
#if USE_MMAP
                    rmpack->unmapFile("lmhead");
#else
                    rmpack->freeMemory("lmhead");
#endif
                    auto input_tensor = external_lmhead_interpretor->getSessionInput(external_lmhead_mnn_session, "in");
                    std::vector<int> input_shape = {1, static_cast<int>(hidden_size)};
                    external_lmhead_interpretor->resizeTensor(input_tensor, input_shape);
                    external_lmhead_interpretor->resizeSession(external_lmhead_mnn_session);
                } catch (const std::exception& e) {
                    LOGE("Failed to load external lmhead: %s", e.what());
                    return RWKV_ERROR_MODEL;
                }
#else
                LOGE("TODO: MNN Windows arm64 building");
                return RWKV_ERROR_MODEL;
#endif
            } else {
                LOGE("Unsupported external lmhead filetype: %s", external_lmhead_filetype.c_str());
                return RWKV_ERROR_MODEL;
            }
        }
    }
    return RWKV_SUCCESS;
}

void qnn_backend::fill_quantized_tensor(float value, Qnn_Tensor_t *tensor) {
    std::vector<size_t> dims;
    for (int j = 0; j < QNN_TENSOR_GET_RANK(*tensor); j++) {
        dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(*tensor) + j));
    }
    void *buffer = qnnIOTensorUtils->getBuffer(tensor);
    float fpzero = 0.0;
    auto dtype = QNN_TENSOR_GET_DATA_TYPE(*tensor);
    if (dtype == QNN_DATATYPE_UFIXED_POINT_8) {
        uint8_t qtzero = 0;
        datautil::floatToTfN<uint8_t>(&qtzero, &fpzero,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
            1);
        for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
            ((uint8_t*)buffer)[j] = qtzero;
        }
    } else if (dtype == QNN_DATATYPE_UFIXED_POINT_16) {
        uint16_t qtzero = 0;
        datautil::floatToTfN<uint16_t>(&qtzero, &fpzero,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
            1);
        for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
            ((uint16_t*)buffer)[j] = qtzero;
        }
    }
}

int qnn_backend::qnn_initialize_tensors() {
    if (!isTensorInitialized) {
        qnnIOTensorUtils->initialize(qnnContextHandles[0]);
        decodeGraphsTensorNameToTensorPointer.resize(qnnDecodeGraphsCount);
        decodeGraphsTensorNameToSize.resize(qnnDecodeGraphsCount);
        if (qnnPrefillGraphsCount > 0) {
            prefillGraphsTensorNameToTensorPointer.resize(qnnPrefillGraphsCount);
            prefillGraphsTensorNameToSize.resize(qnnPrefillGraphsCount);
        }
        if (qnnEmbdGraphsCount > 0) {
            embdGraphsTensorNameToTensorPointer.resize(qnnEmbdGraphsCount);
            embdGraphsTensorNameToSize.resize(qnnEmbdGraphsCount);
        }
        if (qnnEmbdPrefillGraphsCount > 0) {
            embdPrefillGraphsTensorNameToTensorPointer.resize(qnnEmbdPrefillGraphsCount);
            embdPrefillGraphsTensorNameToSize.resize(qnnEmbdPrefillGraphsCount);
        }

        for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
            auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];
            LOGD("Graph %d : %s", graph_id, graphInfo.graphName);

            for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                size_t tensorDataSize = 1;
                for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
                    tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
                }
                auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
                if (typeSize == 0) {
                    return RWKV_ERROR_IO;
                }
                tensorDataSize *= typeSize;
                decodeGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                // LOGI("Output Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);
            }

            if (!qnnIOTensorUtils->setupOutputTensors(&outputTensors[graph_id], decodeGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                          decodeGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], false)) {
                LOGE("Error in setting up Output Tensors");
                return RWKV_ERROR_IO;
            }

            for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                if (tensorName == "out") {
                    logitsOutputTensor = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
                } else if (graph_id == qnnDecodeGraphsCount - 1 && tensorName.find("out_chunk") != std::string::npos) {
                    logitsOutputTensor = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
                }
            }

            std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
            for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                size_t tensorDataSize = 1;
                for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
                    tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
                }
                auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
                size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
                if (typeSize == 0) {
                    return RWKV_ERROR_IO;
                }
                tensorDataSize *= typeSize;
                decodeGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                // LOGI("Input Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);
                if (tensorName.find("state") != std::string::npos) {
                    sharedTensorMap[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName.substr(0, tensorName.find("_in")) + "_out"];
                }
                if (graph_id > 0) {
                    if (tensorName.find("v_first_in") != std::string::npos) {
                        sharedTensorMap[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[0]["v_first_out_chunk1"];
                    } else if (tensorName == "in_chunk" + std::to_string(graph_id + 1)) {
                        sharedTensorMap[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id - 1]["out_chunk" + std::to_string(graph_id)];
                    }
                }
            }

            if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensors[graph_id], decodeGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                        decodeGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                LOGE("Error in setting up Input Tensors");
                return RWKV_ERROR_IO;
            }

            for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                // fill state tensors with zeros
                std::vector<size_t> dims;
                for (int j = 0; j < QNN_TENSOR_GET_RANK(outputTensors[graph_id][i]); j++) {
                    dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(outputTensors[graph_id][i]) + j));
                }
                void *buffer = qnnIOTensorUtils->getBuffer(&outputTensors[graph_id][i]);
                auto tensorName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
                int state_num = 0;
                if (tensorName.find("state") != std::string::npos) {
                    tensorName = std::stoi(tensorName.substr(0, tensorName.find("_")).substr(5));
                }

                if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
                    memset(buffer, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
                else if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
                    memset(buffer, 0, datautil::calculateElementCount(dims) * sizeof(float));
                else {
                    fill_quantized_tensor(0.0, &outputTensors[graph_id][i]);
                }
            }
        }

        if (qnnPrefillGraphsCount > 0) {
            std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMapPrefill;
            for (int graph_id = 0; graph_id < qnnPrefillGraphsCount; graph_id++) {
                auto graphInfo     = (*qnnPrefillGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    prefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Output Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);

                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    } else if (tensorName == "out_prefill") {
                        sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
                    } else if (graph_id == qnnPrefillGraphsCount - 1 && tensorName.find("out_prefill_chunk") != std::string::npos) {
                        sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out_chunk" + std::to_string(graph_id+1)];
                    }
                }

                for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    prefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Input Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);

                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    }

                    if (graph_id > 0) {
                        if (tensorName.find("v_first_in") != std::string::npos) {
                            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[0]["v_first_out_prefill_chunk1"];
                        } else if (tensorName == "in_prefill_chunk" + std::to_string(graph_id + 1)) {
                            sharedTensorMapPrefill[tensorName] = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[graph_id - 1]["out_prefill_chunk" + std::to_string(graph_id)];
                        }
                    }
                }

                if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensorsPrefill[graph_id], prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                prefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapPrefill)) {
                    LOGE("Error in setting up Output Tensors");
                    return RWKV_ERROR_IO;
                }

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsPrefill[graph_id], prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                prefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapPrefill)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }

            }

            Qnn_Tensor_t *tensor = nullptr;
            if (prefillGraphsTensorNameToTensorPointer[0].find("in_prefill") != prefillGraphsTensorNameToTensorPointer[0].end()) {
                tensor = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[0]["in_prefill"];
            } else if (prefillGraphsTensorNameToTensorPointer[0].find("in_prefill_chunk1") != prefillGraphsTensorNameToTensorPointer[0].end()) {
                tensor = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[0]["in_prefill_chunk1"];
            }
            prefillSequenceLength = 1;
            for (int i = 0; i < QNN_TENSOR_GET_RANK(tensor); i++) {
                prefillSequenceLength *= *(QNN_TENSOR_GET_DIMENSIONS(tensor) + i);
            }
            LOGI("Prefill sequence length: %d", prefillSequenceLength);
            tokenInputTensorPrefill = tensor;
        }

        if (qnnEmbdGraphsCount > 0) {
            std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMapEmbd;
            for (int graph_id = 0; graph_id < qnnEmbdGraphsCount; graph_id++) {
                auto graphInfo     = (*qnnEmbdGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    embdGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Output Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);

                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapEmbd[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    } else if (tensorName == "out") {
                        sharedTensorMapEmbd[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
                    }
                }

                for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    embdGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Input Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);
                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapEmbd[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    }
                }

                if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensorsEmbd[graph_id], embdGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapEmbd)) {
                    LOGE("Error in setting up Output Tensors");
                    return RWKV_ERROR_IO;
                }

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsEmbd[graph_id], embdGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapEmbd)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }
            }

            Qnn_Tensor_t *tensor = nullptr;
            if (embdGraphsTensorNameToTensorPointer[0].find("in_embedding") != embdGraphsTensorNameToTensorPointer[0].end()) {
                tensor = (Qnn_Tensor_t*)embdGraphsTensorNameToTensorPointer[0]["in_embedding"];
            }
            tokenInputTensorEmbd = tensor;
        }

        if (qnnEmbdPrefillGraphsCount > 0) {
            if (qnnEmbdPrefillGraphsCount != 1) {
                LOGE("qnnEmbdPrefillGraphsCount: %d, should be 1", qnnEmbdPrefillGraphsCount);
                return RWKV_ERROR_IO;
            }
            std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMapEmbdPrefill;
            for (int graph_id = 0; graph_id < qnnEmbdPrefillGraphsCount; graph_id++) {
                auto graphInfo     = (*qnnEmbdPrefillGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.outputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.outputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    embdPrefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Output Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.outputTensors[i]), tensorDataSize);

                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapEmbdPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    } else if (tensorName == "out_prefill") {
                        sharedTensorMapEmbdPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id]["out"];
                    }
                }

                for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                    size_t tensorDataSize = 1;
                    for (int j = 0; j < QNN_TENSOR_GET_RANK(graphInfo.inputTensors[i]); j++) {
                        tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(graphInfo.inputTensors[i]) + j);
                    }
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
                    size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]));
                    if (typeSize == 0) {
                        return RWKV_ERROR_IO;
                    }
                    tensorDataSize *= typeSize;
                    embdPrefillGraphsTensorNameToSize[graph_id][tensorName] = tensorDataSize;
                    // LOGI("Input Tensor %zu : %s Type: %d Size: %zu", i, tensorName.c_str(), QNN_TENSOR_GET_DATA_TYPE(graphInfo.inputTensors[i]), tensorDataSize);

                    if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMapEmbdPrefill[tensorName] = (Qnn_Tensor_t*)decodeGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    }

                }

                if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensorsEmbdPrefill[graph_id], embdPrefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdPrefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapEmbdPrefill)) {
                    LOGE("Error in setting up Output Tensors");
                    return RWKV_ERROR_IO;
                }

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsEmbdPrefill[graph_id], embdPrefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdPrefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMapEmbdPrefill)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }

            }

            Qnn_Tensor_t *tensor = nullptr;
            if (embdPrefillGraphsTensorNameToTensorPointer[0].find("in_embedding_prefill") != embdPrefillGraphsTensorNameToTensorPointer[0].end()) {
                tensor = (Qnn_Tensor_t*)embdPrefillGraphsTensorNameToTensorPointer[0]["in_embedding_prefill"];
            }
            if (QNN_TENSOR_GET_RANK(tensor) == 2) {
                embdPrefillSequenceLength = *(QNN_TENSOR_GET_DIMENSIONS(tensor));
            } else {
                embdPrefillSequenceLength = *(QNN_TENSOR_GET_DIMENSIONS(tensor) + 1);
            }
            LOGI("Embedding Prefill sequence length: %d", embdPrefillSequenceLength);
            tokenInputTensorEmbdPrefill = tensor;
        }

        isTensorInitialized = true;
    }

    return RWKV_SUCCESS;
}

int qnn_backend::eval(int id, float *& logits) {
    if (!isTensorInitialized) {
        LOGD("qnn_backend::eval() isTensorInitialized: %d", isTensorInitialized);
        return RWKV_ERROR_EVAL;
    }
    int *token_input = (int*)qnnIOTensorUtils->getBuffer(&inputTensors[0][0]);
    *token_input = id;

    for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];
        auto executeStatus =
            qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputTensors[graph_id],
                                                            graphInfo.numInputTensors,
                                                            outputTensors[graph_id],
                                                            graphInfo.numOutputTensors,
                                                            nullptr,
                                                            nullptr);

        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            return RWKV_ERROR_IO;
        }
    }

    // copy logits
    if (logits_buffer.empty()) logits_buffer.resize(vocab_size);
    void *buffer = qnnIOTensorUtils->getBuffer(logitsOutputTensor);

    if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_32) {
        logits = (float*)buffer;
    } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)buffer;
        for (int i = 0; i < vocab_size; i++) {
            logits_buffer[i] = ptr[i];
        }
        logits = logits_buffer.data();
    } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        datautil::tfNToFloat<uint16_t>(logits_buffer.data(), reinterpret_cast<uint16_t*>(buffer),
            QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.scale,
            vocab_size);
        logits = logits_buffer.data();
    } else {
        LOGE("Unsupported data type");
        return RWKV_ERROR_IO;
    }

    return RWKV_SUCCESS;
}

int qnn_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    if (ids.empty()) return RWKV_ERROR_EVAL;
    if (prefillSequenceLength == 0) {
        for (auto id : ids) {
            if (RWKV_SUCCESS != eval(id, logits)) {
                return RWKV_ERROR_MODEL;
            }
        }
    } else {
        int *token_input = (int*)qnnIOTensorUtils->getBuffer(tokenInputTensorPrefill);
        int idx = 0;

        bool is_prefilling_usable = true;
        auto start = std::chrono::high_resolution_clock::now();
        for (; (idx + prefillSequenceLength) <= ids.size(); idx += prefillSequenceLength) {
            for (int i = 0; i < prefillSequenceLength; i++) {
                token_input[i] = ids[idx + i];
            }
            // LOGD("Prefilling using seq mode from %d to %d", idx, idx + prefillSequenceLength);

            for (int graph_id = 0; graph_id < qnnPrefillGraphsCount; graph_id++) {
                auto graphInfo     = (*qnnPrefillGraphsInfo)[graph_id];
                auto executeStatus =
                    qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                    inputTensorsPrefill[graph_id],
                                                                    graphInfo.numInputTensors,
                                                                    outputTensorsPrefill[graph_id],
                                                                    graphInfo.numOutputTensors,
                                                                    nullptr,
                                                                    nullptr);

                if (QNN_GRAPH_NO_ERROR != executeStatus) {
                    is_prefilling_usable = false;
                    LOGE("QNN: prefill graph not usable; falling back to decode mode");
                    break;
                }
            }
            if (!is_prefilling_usable) {
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        prefill_speed = (ids.size() / prefillSequenceLength * prefillSequenceLength) * 1000000.0 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // LOGD("Prefilling tails using decode mode from %d to %d", idx, ids.size());
        for (; idx < ids.size(); idx++) {
            token_input = (int*)qnnIOTensorUtils->getBuffer(&inputTensors[0][0]);
            *token_input = ids[idx];
            for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
                auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];
                auto executeStatus =
                    qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                    inputTensors[graph_id],
                                                                    graphInfo.numInputTensors,
                                                                    outputTensors[graph_id],
                                                                    graphInfo.numOutputTensors,
                                                                    nullptr,
                                                                    nullptr);

                if (QNN_GRAPH_NO_ERROR != executeStatus) {
                    return RWKV_ERROR_IO;
                }
            }
        }
    }

    // copy logits
    if (!skip_logits_copy) {
        if (logits_buffer.empty()) logits_buffer.resize(vocab_size);
        void *buffer = qnnIOTensorUtils->getBuffer(logitsOutputTensor);

        if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_32) {
            logits = (float*)buffer;
        } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_16) {
            half_float::half *ptr = (half_float::half*)buffer;
            for (int i = 0; i < vocab_size; i++) {
                logits_buffer[i] = ptr[i];
            }
            logits = logits_buffer.data();
        } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_UFIXED_POINT_16) {
            datautil::tfNToFloat<uint16_t>(logits_buffer.data(), reinterpret_cast<uint16_t*>(buffer),
                QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.offset,
                QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.scale,
                vocab_size);
            logits = logits_buffer.data();
        } else {
            LOGE("Unsupported data type");
            return RWKV_ERROR_IO;
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::eval_with_embeddings(const float *embeddings, int n_tokens, float *& logits) {
    if (!isTensorInitialized) return RWKV_ERROR_EVAL;
    LOGD("[QNN] eval_with_embeddings: n_tokens: %d", n_tokens);

    int i = 0;
    if (embdPrefillSequenceLength > 0) {
        void *embedding_input_prefill = qnnIOTensorUtils->getBuffer(tokenInputTensorEmbdPrefill);
        for (; i + embdPrefillSequenceLength <= n_tokens; i += embdPrefillSequenceLength) {
            if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbdPrefill) == QNN_DATATYPE_FLOAT_32) {
                memcpy(embedding_input_prefill, (float*)(embeddings + i * hidden_size), embdPrefillSequenceLength * hidden_size * sizeof(float));
            } else if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbdPrefill) == QNN_DATATYPE_FLOAT_16) {
                half_float::half *ptr = (half_float::half*)embedding_input_prefill;
                for (int j = 0; j < embdPrefillSequenceLength * hidden_size; j++) {
                    ptr[j] = (half_float::half)(embeddings[i * hidden_size + j]);
                }
            } else if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbdPrefill) == QNN_DATATYPE_UFIXED_POINT_16) {
                datautil::floatToTfN<uint16_t>(reinterpret_cast<uint16_t*>(embedding_input_prefill), (float*)(embeddings + i * hidden_size),
                        QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbdPrefill).scaleOffsetEncoding.offset,
                        QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbdPrefill).scaleOffsetEncoding.scale,
                        embdPrefillSequenceLength * hidden_size);
            } else {
                LOGE("Unsupported data type");
                return RWKV_ERROR_IO;
            }

            auto graphInfo = (*qnnEmbdPrefillGraphsInfo)[0];
            auto executeStatus =
                qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                inputTensorsEmbdPrefill[0],
                                                                graphInfo.numInputTensors,
                                                                outputTensorsEmbdPrefill[0],
                                                                graphInfo.numOutputTensors,
                                                                nullptr, nullptr);
            if (QNN_GRAPH_NO_ERROR != executeStatus) {
                return RWKV_ERROR_IO;
            }
        }
    }

    // leftovers
    for (; i < n_tokens; i++) {
        void *embedding_input = qnnIOTensorUtils->getBuffer(tokenInputTensorEmbd);
        if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbd) == QNN_DATATYPE_FLOAT_32) {
            memcpy(embedding_input, (float*)(embeddings + i * hidden_size), hidden_size * sizeof(float));
        } else if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbd) == QNN_DATATYPE_FLOAT_16) {
            half_float::half *ptr = (half_float::half*)embedding_input;
            for (int j = 0; j < hidden_size; j++) {
                ptr[j] = (half_float::half)(embeddings[i * hidden_size + j]);
            }
        } else if (QNN_TENSOR_GET_DATA_TYPE(tokenInputTensorEmbd) == QNN_DATATYPE_UFIXED_POINT_16) {
            datautil::floatToTfN<uint16_t>(reinterpret_cast<uint16_t*>(embedding_input), (float*)(embeddings + i * hidden_size),
                    QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbd).scaleOffsetEncoding.offset,
                    QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbd).scaleOffsetEncoding.scale,
                    hidden_size);
        } else {
            LOGE("Unsupported data type");
            return RWKV_ERROR_IO;
        }

        auto graphInfo = (*qnnEmbdGraphsInfo)[0];
        auto executeStatus =
            qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputTensorsEmbd[0],
                                                            graphInfo.numInputTensors,
                                                            outputTensorsEmbd[0],
                                                            graphInfo.numOutputTensors,
                                                            nullptr, nullptr);
        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            return RWKV_ERROR_IO;
        }
    }

    if (logits_buffer.empty()) logits_buffer.resize(vocab_size);
    void *buffer = qnnIOTensorUtils->getBuffer(logitsOutputTensor);

    if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_32) {
        logits = (float*)buffer;
    } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)buffer;
        for (int i = 0; i < vocab_size; i++) {
            logits_buffer[i] = ptr[i];
        }
    } else if (QNN_TENSOR_GET_DATA_TYPE(logitsOutputTensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        datautil::tfNToFloat<uint16_t>(logits_buffer.data(), reinterpret_cast<uint16_t*>(buffer),
            QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(logitsOutputTensor).scaleOffsetEncoding.scale,
            vocab_size);
    } else {
        LOGE("Unsupported data type");
        return RWKV_ERROR_IO;
    }
    logits = logits_buffer.data();

    return RWKV_SUCCESS;
}
bool qnn_backend::is_available() {
    // TODO: Detect this
    return true;
}

int qnn_backend::clear_state() {
    if (!isTensorInitialized) return RWKV_SUCCESS;
    for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            size_t element_count = 1;
            for (int j = 0; j < QNN_TENSOR_GET_RANK(outputTensors[graph_id][i]); j++) {
                element_count *= *(QNN_TENSOR_GET_DIMENSIONS(outputTensors[graph_id][i]) + j);
            }
            void *buffer = qnnIOTensorUtils->getBuffer(&outputTensors[graph_id][i]);
            if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
                memset(buffer, 0, element_count * sizeof(uint16_t));
            else if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
                memset(buffer, 0, element_count * sizeof(float));
            else {
                fill_quantized_tensor(0.0, &outputTensors[graph_id][i]);
            }
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::get_state(std::any &state) {
    auto new_state = std::vector<std::vector<uint8_t>>();
    for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
            
            if (outputName.find("state") != std::string::npos) {
                new_state.push_back(std::vector<uint8_t>((uint8_t*)qnnIOTensorUtils->getBuffer(&outputTensors[graph_id][i]), (uint8_t*)qnnIOTensorUtils->getBuffer(&outputTensors[graph_id][i]) + qnnIOTensorUtils->getBufferSize(&outputTensors[graph_id][i])));
            }
        }
    }
    state = new_state;
    return RWKV_SUCCESS;
}

int qnn_backend::set_state(std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    int idx = 0;
    for (int graph_id = 0; graph_id < qnnDecodeGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnDecodeGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
            
            if (outputName.find("state") != std::string::npos) {
                memcpy(qnnIOTensorUtils->getBuffer(&outputTensors[graph_id][i]), new_state[idx].data(), new_state[idx].size());
                idx++;
            }
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::free_state(std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (auto &s : new_state) {
        s.clear();
    }
    new_state.clear();
    return RWKV_SUCCESS;
}

int qnn_backend::release_model() {
    // free graphs
    for (int i = 0; i < qnnDecodeGraphsCount; i++) {
        auto graphInfo     = (*qnnDecodeGraphsInfo)[i];
        qnnIOTensorUtils->tearDownTensors(inputTensors[i], graphInfo.numInputTensors);
        qnnIOTensorUtils->tearDownTensors(outputTensors[i], graphInfo.numOutputTensors);
        inputTensors[i]  = nullptr;
        outputTensors[i] = nullptr;
    }

    freeGraphsInfo(&qnnDecodeGraphsInfo, qnnDecodeGraphsCount);
    qnnDecodeGraphsInfo = nullptr;

    if (qnnPrefillGraphsCount > 0) {
        for (int i = 0; i < qnnPrefillGraphsCount; i++) {
            auto graphInfo     = (*qnnPrefillGraphsInfo)[i];
            qnnIOTensorUtils->tearDownTensors(inputTensorsPrefill[i], graphInfo.numInputTensors);
            qnnIOTensorUtils->tearDownTensors(outputTensorsPrefill[i], graphInfo.numOutputTensors);
            inputTensorsPrefill[i]  = nullptr;
            outputTensorsPrefill[i] = nullptr;
        }

        freeGraphsInfo(&qnnPrefillGraphsInfo, qnnPrefillGraphsCount);
        qnnPrefillGraphsInfo = nullptr;
    }

    if (qnnEmbdGraphsCount > 0) {
        for (int i = 0; i < qnnEmbdGraphsCount; i++) {
            auto graphInfo     = (*qnnEmbdGraphsInfo)[i];
            qnnIOTensorUtils->tearDownTensors(inputTensorsEmbd[i], graphInfo.numInputTensors);
            qnnIOTensorUtils->tearDownTensors(outputTensorsEmbd[i], graphInfo.numOutputTensors);
            inputTensorsEmbd[i]  = nullptr;
            outputTensorsEmbd[i] = nullptr;
        }

        freeGraphsInfo(&qnnEmbdGraphsInfo, qnnEmbdGraphsCount);
        qnnEmbdGraphsInfo = nullptr;
    }

    for (int i = 0; i < qnnContextHandles.size(); i++) {
        if (QNN_CONTEXT_NO_ERROR !=
            qnnFunctionPointers.qnnInterface.contextFree(qnnContextHandles[i], nullptr)) {
            LOGE("Could not free context");
        }
    }

    qnn_destory_power_config_id();

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }

        auto qnnStatus = qnnFunctionPointers.qnnInterface.deviceFree(qnnDeviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
            LOGE("Failed to free device");
        }
    }

    if (qnnBackendLibraryHandle)
        pal::dynamicloading::dlClose(qnnBackendLibraryHandle);

    if (qnnModelHandle)
        pal::dynamicloading::dlClose(qnnModelHandle);

    for (int i = 0; i < graphConfigsInfoCount; i++) {
        delete graphConfigsInfo[i];
    }
    delete graphConfigsInfo;

    if ((nullptr != qnnBackendHandle && nullptr != qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendFree(qnnBackendHandle)) {
        LOGE("Could not terminate QNN backend");
    }
    qnnBackendHandle = nullptr;

    if (nullptr != qnnFunctionPointers.qnnInterface.logFree && nullptr != qnnLogHandle) {
        if (QNN_SUCCESS != qnnFunctionPointers.qnnInterface.logFree(qnnLogHandle)) {
            LOGW("Unable to terminate logging in the backend.");
        }
    }

    return RWKV_SUCCESS;
}

int qnn_backend::release() {
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_register_op_package(std::string package_path, std::string interface_provider) {
    if (nullptr == qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
        LOGE("backendRegisterOpPackageFnHandle is nullptr.");
        return RWKV_ERROR_UNSUPPORTED;
    }
    if (QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                qnnBackendHandle,
                package_path.c_str(),
                interface_provider.c_str(),
                nullptr)) {
        LOGE("Could not register Op Package: %s and interface provider: %s",
            package_path.c_str(),
            interface_provider.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    LOGI("Registered Op Package: %s and interface provider: %s",
        package_path.c_str(),
        interface_provider.c_str()
    );
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_create_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr = perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      LOGE("createPowerConfigId failed");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_destory_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.destroyPowerConfigId(powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("destroyPowerConfigId failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_set_power_config() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("device error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable    = 0; //True to enable Dcvs, False to disbale
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId     = powerConfigId;  //use the power config id created

    // refer QnnHtpPerfInfrastructure.h
    powerConfig.dcvsV3Config.powerMode       = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1; //True to consider Latency parameter otherwise False
    powerConfig.dcvsV3Config.setBusParams    = 1; //True to consider Bus parameter otherwise False
    powerConfig.dcvsV3Config.setCoreParams   = 1; //True to consider Core parameter otherwise False
    powerConfig.dcvsV3Config.sleepDisable    = 1; //True to disable sleep, False to re-enable sleep
    powerConfig.dcvsV3Config.setSleepDisable = 1; //True to consider sleep disable/enable parameter otherwise False

    //Set Sleep latency parameter
    powerConfig.dcvsV3Config.sleepLatency    =  40; // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk

    //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfigHMX;
    memset(&powerConfigHMX, 0, sizeof(powerConfigHMX));
    powerConfigHMX.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
    powerConfigHMX.hmxV2Config.hmxPickDefault = 0;
    powerConfigHMX.hmxV2Config.hmxPerfMode    = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

    powerConfigHMX.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_TUR;

    // Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, &powerConfigHMX, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (perfInfraErr != QNN_SUCCESS) {
        const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigsWithoutHMX[] = {&powerConfig, NULL};
        perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigsWithoutHMX);
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_set_rpc_latency_and_polling() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      LOGE("device error");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;            // refer QnnHtpPerfInfrastructure.h
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;         // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);  // set RPC latency config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;   // refer QnnHtpPerfInfrastructure.h
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;     // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, NULL};

    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2); // set RPC polling config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

#else

int qnn_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::eval(int id, float *& logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::clear_state() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::get_state(std::any &state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::set_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::free_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

bool qnn_backend::is_available() {
    return false;
}

int qnn_backend::release_model() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::release() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

#endif

} // namespace rwkvmobile