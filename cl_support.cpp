#include "cl_support.hpp"

#include <vector>
#include <regex>
#include <exception>

cl::Device GetDevice(cl_device_type type_mask, std::string filter_regex) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::regex rex {filter_regex};
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(type_mask, &devices);
        for (auto& d : devices) {
            std::smatch r;
            auto name = d.getInfo<CL_DEVICE_NAME>();
            if (std::regex_match(name, r, rex)) {
                return d;
            }
        }
    }
    throw std::runtime_error{"No device matched"};
}

cl_device_type ClTypeMaskFromString(const std::string& s) {
    cl_device_type ret {};
    if (s.find("default") != std::string::npos) {
        ret |= CL_DEVICE_TYPE_DEFAULT;
    }
    if (s.find("cpu") != std::string::npos) {
        ret |= CL_DEVICE_TYPE_CPU;
    }
    if (s.find("gpu") != std::string::npos) {
        ret |= CL_DEVICE_TYPE_GPU;
    }
    if (s.find("accelerator") != std::string::npos) {
        ret |= CL_DEVICE_TYPE_ACCELERATOR;
    }
    if (s.find("all") != std::string::npos) {
        ret |= CL_DEVICE_TYPE_ALL;
    }
    return ret;
}
