#include <iostream>
#include "src/runtime.h"
#include "src/commondef.h"

int main() {
    std::cout << "Testing OpenCog RWKV Backend Integration" << std::endl;
    
    rwkvmobile::runtime runtime;
    
    // Get available backends
    std::vector<int> backend_ids;
    int ret = runtime.get_available_backend_ids(backend_ids);
    
    if (ret != rwkvmobile::RWKV_SUCCESS) {
        std::cout << "Failed to get available backends" << std::endl;
        return 1;
    }
    
    std::cout << "Available backends: " << runtime.get_available_backends_str() << std::endl;
    
    // Check if OpenCog backend is available
    bool opencog_found = false;
    for (int id : backend_ids) {
        if (id == rwkvmobile::RWKV_BACKEND_OPENCOG) {
            opencog_found = true;
            break;
        }
    }
    
    if (opencog_found) {
        std::cout << "✓ OpenCog backend is available!" << std::endl;
        
        // Test backend string conversion
        if (rwkvmobile::backend_str_to_enum("opencog") == rwkvmobile::RWKV_BACKEND_OPENCOG) {
            std::cout << "✓ Backend string conversion works!" << std::endl;
        } else {
            std::cout << "✗ Backend string conversion failed!" << std::endl;
        }
        
        if (rwkvmobile::backend_enum_to_str(rwkvmobile::RWKV_BACKEND_OPENCOG) == "opencog") {
            std::cout << "✓ Backend enum conversion works!" << std::endl;
        } else {
            std::cout << "✗ Backend enum conversion failed!" << std::endl;
        }
        
    } else {
        std::cout << "✗ OpenCog backend is not available" << std::endl;
        return 1;
    }
    
    std::cout << "OpenCog RWKV backend integration test completed successfully!" << std::endl;
    return 0;
}