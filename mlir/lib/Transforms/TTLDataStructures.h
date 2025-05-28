#ifndef TTL_DATA_STRUCTURES_H
#define TTL_DATA_STRUCTURES_H

#include <string>
#include <vector>
#include <map>
#include <set>

namespace mlir {

// Represents a loop variable's properties
struct LoopVariable {
    int start;
    int end;
    int step;
    int current;
    bool isDynamic;
    
    LoopVariable() : start(0), end(0), step(0), current(0), isDynamic(false) {}
};

// Represents tensor access information
struct TensorAccess {
    int rank;
    int x_stride;
    int y_stride;
    int z_stride;
    std::string x_index;  // Can be "x", "y", "z" or "dynamic"
    std::string y_index;
    std::string z_index;
    bool isDynamic;
    
    TensorAccess() : rank(0), x_stride(0), y_stride(0), z_stride(0), 
                     x_index("dynamic"), y_index("dynamic"), z_index("dynamic"),
                     isDynamic(false) {}
};

// Represents a complete loop structure with its accessed tensors
struct LoopInfo {
    std::string name;  // x, y, or z
    LoopVariable var;
    std::set<std::string> accessedTensors;  // Names of tensors accessed in this loop
    
    LoopInfo(const std::string& n) : name(n) {}
    
    void addTensorAccess(const std::string& tensorName) {
        accessedTensors.insert(tensorName);
    }
};

// Main structure to hold all analysis information
struct TTLAnalysis {
    std::vector<LoopInfo> loops;
    std::map<std::string, TensorAccess> tensorAccesses;  // Key is tensor name (e.g., "Tensor1")
    
    // Helper methods
    void addLoop(const std::string& name, int start, int end, int step, int current, bool isDynamic) {
        LoopInfo loop(name);
        loop.var.start = start;
        loop.var.end = end;
        loop.var.step = step;
        loop.var.current = current;
        loop.var.isDynamic = isDynamic;
        loops.push_back(loop);
    }
    
    void updateLoopEnd(const std::string& name, int end, bool isDynamic) {
        for (auto& loop : loops) {
            if (loop.name == name) {
                loop.var.end = end;
                loop.var.isDynamic = isDynamic;
                break;
            }
        }
    }
    
    void updateLoopStep(const std::string& name, int step, bool isDynamic) {
        for (auto& loop : loops) {
            if (loop.name == name) {
                loop.var.step = step;
                loop.var.isDynamic = isDynamic;
                break;
            }
        }
    }
    
    void updateLoopCurrent(const std::string& name, int current, bool isDynamic) {
        for (auto& loop : loops) {
            if (loop.name == name) {
                loop.var.current = current;
                loop.var.isDynamic = isDynamic;
                break;
            }
        }
    }
    
    void addTensorAccess(const std::string& name, const TensorAccess& access) {
        tensorAccesses[name] = access;
    }
    
    // Add tensor access to a specific loop
    void addTensorAccessToLoop(const std::string& loopName, const std::string& tensorName) {
        for (auto& loop : loops) {
            if (loop.name == loopName) {
                loop.addTensorAccess(tensorName);
                break;
            }
        }
    }
    
    // Get all tensors accessed in a specific loop
    std::set<std::string> getTensorsInLoop(const std::string& loopName) const {
        for (const auto& loop : loops) {
            if (loop.name == loopName) {
                return loop.accessedTensors;
            }
        }
        return {};
    }
    
    // Get all loops that access a specific tensor
    std::vector<std::string> getLoopsForTensor(const std::string& tensorName) const {
        std::vector<std::string> result;
        for (const auto& loop : loops) {
            if (loop.accessedTensors.count(tensorName) > 0) {
                result.push_back(loop.name);
            }
        }
        return result;
    }
};

} // namespace mlir

#endif // TTL_DATA_STRUCTURES_H 