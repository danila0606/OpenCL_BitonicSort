#include "BitonicSorter.h"
#include <fstream>
#include <cmath>

namespace BS {
    BitonicSorter::BitonicSorter(const std::string &kernel_name) {


        // ChoosePlatformAndDevice();
        ChooseDefaultPlatformAndDevice();

        work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

        context_ = cl::Context({device_});
        queue_ = cl::CommandQueue(context_, device_);

        std::ifstream program_sources(kernel_name);
        std::string program_string(std::istreambuf_iterator<char>(program_sources), (std::istreambuf_iterator<char>()));
        source_ = cl::Program::Sources(1, std::make_pair(program_string.c_str(), program_string.length() + 1));

        program_ = cl::Program(context_, source_);

        program_.build();
    }

    long int BitonicSorter::Sort(std::vector<int> &data, Dir direction) {

        int dir = static_cast<int>(direction);

        cl::Kernel sort = cl::Kernel(program_, "bitonic_sort_kernel_local");

        //preparing data
        size_t old_size = data.size();
        size_t new_size = 1;
        while (new_size < old_size)
            new_size *= 2;

        int pushing_num = 0;
        if (dir)
            pushing_num = std::numeric_limits<int>::max();
        else
            pushing_num = std::numeric_limits<int>::min();

        data.reserve(new_size);
        for (size_t i = old_size; i < new_size; ++i)
            data.push_back(pushing_num);
        //preparing data


        size_t global_size = data.size() / 2;
        // This is the size of the work group.
        size_t local_size = std::min(global_size, work_group_size);
        // Calculate the Number of work groups.

        auto start_t = std::chrono::system_clock::now();

        cl::Buffer buffer(context_, CL_MEM_READ_WRITE, sizeof(int) * data.size());
        queue_.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(int) * data.size(), data.data());

        unsigned numStages = std::ceil(std::log2(new_size));
        unsigned localStages = std::log2(local_size);

        // While local_size < work_group_size we can use local memory.
        // After this we must use default kernel, without using local memory.
        cl::LocalSpaceArg local = cl::Local(2 * local_size * sizeof(int));

        sort.setArg(0, buffer);
        sort.setArg(1, local);
        sort.setArg(2, localStages);
        sort.setArg(3, dir);

        cl::Event event;
        queue_.enqueueNDRangeKernel(sort, 0, global_size, local_size, nullptr, &event);
        event.wait();


        sort = cl::Kernel(program_, "bitonic_sort_kernel_default");

        int curStage = localStages;
        for (; curStage < numStages; ++curStage) {
            sort.setArg(0, buffer);
            sort.setArg(1, curStage);

            // Every stage has stage + 1 passes
            for (int passOfStage = 0; passOfStage < curStage + 1; ++passOfStage) {

                sort.setArg(2, passOfStage);
                sort.setArg(3, dir);

                // Enqueue a kernel run call.
                // Each thread writes a sorted pair.
                // So, the number of threads (global) should be half the length of the input buffer.

                cl::Event event1;
                queue_.enqueueNDRangeKernel(sort, cl::NullRange, global_size, local_size, nullptr, &event1);
                event1.wait();

                queue_.finish();
            }
        }

        queue_.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int) * data.size(), data.data());
        data.resize(old_size);
        auto finish_t = std::chrono::system_clock::now();

        return (finish_t - start_t).count();
    }

    void BitonicSorter::ChoosePlatformAndDevice() {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cout << " No platforms found\n";
            exit(1);
        }
        int size = platforms.size();

        //Get all devices for each platform
        std::vector<std::vector<cl::Device>> all_devices(size);
        for (int i = 0; i < size; ++i)
            platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &all_devices[i]);

        //if only one platform available
        if (size == 1) {
            std::cout << "You have one platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

            if (all_devices[0].empty()) {
                std::cerr << "No devices found!" << std::endl;
                exit(1);
            }
            platform_ = platforms[0];

            int devices_count = all_devices[0].size();
            if (devices_count == 1) {
                std::cout << "You have one device: " << all_devices[0][0].getInfo<CL_DEVICE_NAME>() << std::endl;
                device_ = all_devices[0][0];
                return;
            }

            std::cout << "You have " << devices_count << " devices available.\nChoose one (write number)" << std::endl;
            for (int i = 0; i < devices_count; ++i)
                std::cout << "[" << i << "]: " << all_devices[0][i].getInfo<CL_DEVICE_NAME>() << std::endl;

            int number = 0;
            std::cin >> number;
            while (number < 0 && number > devices_count) {
                std::cout << "Wrong number. Try again!" << std::endl;
                std::cin >> number;
            }

            device_ = all_devices[0][number];

            return;
        }

        std::cout << "You have " << size << " platforms available.\nChoose one (write number)" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << "[" << i << "]: " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << "\tDevices available on this platform: " << std::endl;
            for (int j = 0; j < all_devices[i].size(); ++j)
                std::cout << "\t[" << j << "]: " << all_devices[i][j].getInfo<CL_DEVICE_NAME>() << std::endl;

            std::cout << "-------------------------------------\n";
        }

        int number = 0;
        std::cin >> number;
        while (number < 0 && number > size) {
            std::cout << "Wrong number. Try again!" << std::endl;
            std::cin >> number;
        }
        platform_ = platforms[number];

        int devices_count = all_devices[number].size();
        if (!devices_count) {
            std::cerr << "No devices found!" << std::endl;
            exit(1);
        }

        std::cout << "You have " << devices_count << " devices available.\nChoose one (write number)" << std::endl;
        for (int i = 0; i < devices_count; ++i)
            std::cout << "[" << i << "]: " << all_devices[number][i].getInfo<CL_DEVICE_NAME>() << std::endl;

        int N = 0;
        std::cin >> N;
        while (N < 0 && N > devices_count) {
            std::cout << "Wrong number. Try again!" << std::endl;
            std::cin >> N;
        }

        device_ = all_devices[number][N];
    }

    void BitonicSorter::ChooseDefaultPlatformAndDevice() {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty())
            throw std::invalid_argument("No platforms found");


        for (auto &platform : platforms) {

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (!devices.empty()) {
                platform_ = platform;
                device_ = devices[0];
                return;
            }
        }

        throw std::invalid_argument("No devices found");
    }

}
