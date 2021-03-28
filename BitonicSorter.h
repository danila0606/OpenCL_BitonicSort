#pragma once

#include <iostream>
#include <sstream>
#include <chrono>
#include <exception>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <fstream>
#include <cassert>


namespace BS {
    enum class Dir {
        Decrease,
        Increase
    };

    class BitonicSorter final {
    private:

        cl::Platform platform_;
        cl::Context context_;
        cl::Device device_;
        cl::Program::Sources source_;
        cl::CommandQueue queue_;
        cl::Program program_;
        cl::Kernel sort_fast;
        cl::Kernel sort_default;

        const std::string kernel_name_;

    private:
        size_t work_group_size = 0;

    public:

        explicit BitonicSorter(const std::string& kernel_name = "sort_kernel.cl");

        long int Sort(std::vector<int> &data, Dir direction = Dir::Increase); //returns time spent on sorting

        void ChoosePlatformAndDevice(); //choose by user in console
        void ChooseDefaultPlatformAndDevice(); //choose first suited platform and device

    };
}