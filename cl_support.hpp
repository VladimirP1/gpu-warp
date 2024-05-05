#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <string>

#include <CL/opencl.hpp>

cl::Device GetDevice(cl_device_type type_mask, std::string filter_regex);

cl_device_type ClTypeMaskFromString(const std::string &s);
