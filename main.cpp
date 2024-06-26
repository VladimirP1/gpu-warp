#include <algorithm>
#include <cmath>
#include <iostream>

#include "utils.hpp"
#include "cl_support.hpp"

#include <opencv2/opencv.hpp>

typedef struct __attribute__((packed)) _args {
    float transform[6];
    uint in_dim[2];
    uint out_dim[2];
} kernel_args;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Bad argument count" << std::endl;
        std::cerr << "Usage: ./main <device_type> <device_name_regex>" << std::endl;
        std::cerr << "  <device_type> can be: default, gpu, cpu, accelerator, all" << std::endl;
        std::cerr << "  <device_name_regex> is regex the dievice name should match" << std::endl;

        return 1;
    }

    cv::Mat img_fc, img_in = cv::imread("../test.png");
    img_in.convertTo(img_fc, CV_32FC3);

    cv::Mat img_out = cv::Mat::zeros(cv::Size(1920, 1080), CV_32FC3);

    cl::Device dev;
    cl::Context ctx;
    try {
        dev = GetDevice(ClTypeMaskFromString(argv[1]), argv[2]);
        ctx = cl::Context(dev);
    } catch (const cl::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    cl::CommandQueue queue{ctx, dev, cl::QueueProperties::Profiling};

    size_t local_size[] = {8, 8};

    cl::Program prog;
    try {
        std::string kernel_source = read_file("../kernel.cl");
        prog = cl::Program{ctx, kernel_source.c_str()};
        prog.build();
    } catch (const cl::BuildError &e) {
        for (auto &log : e.getBuildLog()) {
            std::cerr << log.second << std::endl;
        }
        throw std::runtime_error{std::string{"Build completed with errors: "} + e.what()};
    } catch (const cl::Error &e) {
        throw std::runtime_error{std::string{"Build failed: "} + e.what()};
    }

    cl::Event evt;
    cl::Kernel kernel{prog, "warp"};

    std::vector<cl_float> in_buf(3 * img_fc.rows * img_fc.cols);
    std::vector<cl_float> out_buf(3 * img_out.rows * img_out.cols);

    cl::Buffer img_in_mem{ctx, CL_MEM_READ_ONLY, in_buf.size() * sizeof(cl_float)};
    cl::Buffer img_out_mem{ctx, CL_MEM_READ_WRITE, out_buf.size() * sizeof(cl_float)};

    std::cout << in_buf.size() << std::endl;

    for (int i = 0; i < img_fc.rows; ++i) {
        for (int j = 0; j < img_fc.cols; ++j) {
            int p = img_fc.cols * 3 * i + 3 * j;
            for (int k = 0; k < 3; ++k) {
                in_buf[p + k] = img_fc.at<cv::Vec3f>(i, j).val[k];
            }
        }
    }

    float ang = 0.1;
    kernel_args args{
        .transform = {2.2f * cos(ang), 2.2f * -sin(ang), 0, 2.2f * sin(ang), 2.2f * cos(ang), 0},
        // .transform = {5, 0, 0, 0, 5, 0},
        .in_dim = {(cl_uint)img_fc.rows, (cl_uint)img_fc.cols},
        .out_dim = {(cl_uint)img_out.rows, (cl_uint)img_out.cols},
    };
    kernel.setArg(0, sizeof(cl_mem) * 1, &img_in_mem);
    kernel.setArg(1, sizeof(cl_mem) * 1, &img_out_mem);
    kernel.setArg(2, sizeof(kernel_args), &args);

    queue.enqueueWriteBuffer(img_in_mem, true, 0, in_buf.size() * sizeof(cl_float), in_buf.data());

    queue.enqueueNDRangeKernel(kernel, cl::NDRange{0, 0},
                               cl::NDRange{(size_t)img_out.rows, (size_t)img_out.cols},
                               cl::NullRange, nullptr, &evt);

    queue.enqueueReadBuffer(img_out_mem, true, 0, out_buf.size() * sizeof(cl_float),
                            out_buf.data());
    queue.flush();

    cl_ulong start, end;
    evt.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    evt.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);

    std::cout << "time " << (end - start) / 1000 << " us" << std::endl;

    for (int i = 0; i < img_out.rows; ++i) {
        for (int j = 0; j < img_out.cols; ++j) {
            int p = img_out.cols * 3 * i + 3 * j;
            for (int k = 0; k < 3; ++k) {
                img_out.at<cv::Vec3f>(i, j).val[k] = out_buf[p + k];
            }
        }
    }

    cv::imwrite("out.png", img_out);
    return 0;
}