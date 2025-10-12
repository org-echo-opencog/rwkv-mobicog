# OpenCog RWKV Backend

## Overview

The OpenCog RWKV backend integrates the RWKV language model with OpenCog's cognitive architecture, enabling symbolic reasoning and cognitive processing capabilities on top of neural language modeling.

## Features

### Core Components

1. **RWKV Atom Types**: Specialized atom types for representing RWKV components in OpenCog's AtomSpace
   - `RWKV_TOKEN_NODE`: Represents vocabulary tokens
   - `RWKV_STATE_NODE`: Represents internal RWKV states
   - `RWKV_LAYER_NODE`: Represents RWKV layers
   - `RWKV_ATTENTION_LINK`: Represents attention relationships
   - `RWKV_SEQUENCE_LINK`: Represents token sequences
   - `RWKV_CONTEXT_LINK`: Represents contextual relationships
   - `RWKV_PREDICTION_LINK`: Represents prediction relationships

2. **Cognitive Processing**: 
   - Pattern matching for sequence similarity
   - Attention-based inference
   - Context understanding through symbolic representation
   - Probabilistic reasoning integration

3. **Semantic Networks**: Automatic construction of semantic relationships from processed sequences

### Architecture

The backend uses a PIMPL (Pointer to Implementation) pattern to encapsulate OpenCog-specific data structures while maintaining compatibility with the existing RWKV mobile framework.

#### Key Classes

- `opencog_rwkv_backend`: Main backend class implementing the execution_provider interface
- `RWKVAtomFactory`: Factory for creating OpenCog atoms from RWKV data
- `RWKVReasoning`: Cognitive reasoning operations on RWKV sequences
- `RWKVCognitiveGraph`: High-level cognitive graph operations

## Usage

### Building with OpenCog Backend

```bash
cmake .. -DENABLE_OPENCOG_BACKEND=ON
make -j$(nproc)
```

### Using the Backend

```cpp
#include "runtime.h"

rwkvmobile::runtime runtime;
int model_id = runtime.load_model("model.bin", "opencog", "vocab.json", nullptr);

// The backend will automatically:
// 1. Create symbolic representations of tokens in AtomSpace
// 2. Build contextual relationships
// 3. Apply cognitive reasoning to enhance predictions
// 4. Learn patterns from processed sequences
```

### Available as Backend

The OpenCog backend is registered as `"opencog"` and can be used anywhere the RWKV mobile framework accepts a backend name.

## Cognitive Enhancements

### 1. Contextual Probability Enhancement

The backend enhances base RWKV probabilities using OpenCog reasoning:
```
enhanced_logit = base_logit + log(contextual_probability)
```

### 2. Sequence Pattern Learning

The system automatically builds semantic networks from processed token sequences, identifying:
- Token co-occurrence patterns  
- Attention relationships
- Conceptual clusters
- Predictive patterns

### 3. Goal-Oriented Generation

The cognitive graph can generate token sequences oriented towards specific goals or concepts represented as OpenCog atoms.

### 4. Memory Consolidation

Episodic memories of processed sequences are consolidated over time, strengthening frequently occurring patterns.

## Implementation Details

### Memory Management
- Uses PIMPL pattern for clean separation of concerns
- Automatic cleanup of OpenCog resources
- Bounded memory usage through sequence history limits

### Performance Considerations
- Cognitive processing is applied selectively (every 10 tokens by default)
- Pattern matching uses efficient indexing
- Memory limits prevent unbounded growth

### Error Handling
- Graceful degradation when cognitive processing fails
- Fallback to base RWKV behavior on errors
- Robust initialization and cleanup

## Future Enhancements

1. **Real OpenCog Integration**: Currently uses simplified implementations; can be extended to use full OpenCog AtomSpace
2. **Advanced Reasoning**: Integration with OpenCog's PLN (Probabilistic Logic Networks)
3. **Multi-Modal Reasoning**: Extension to handle non-textual cognitive processing
4. **Learning Algorithms**: Integration with OpenCog's learning and adaptation mechanisms

## Dependencies

- Standard C++17 library
- Existing RWKV mobile framework dependencies
- No additional external dependencies (uses embedded implementations)

## Files

- `opencog_rwkv_backend.h/cpp`: Main backend implementation
- `rwkv_atoms.h/cpp`: OpenCog atom types and cognitive processing
- `README.md`: This documentation

## License

Same as the parent RWKV mobile project.