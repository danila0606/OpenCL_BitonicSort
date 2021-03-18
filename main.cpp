#include "BitonicSorter.h"
#include <random>

int main(int argc, char* argv[]) {

    std::ifstream in{argv[1]};
    if (!in.is_open()) {
        std::cerr << "Cant open file!" << std::endl;
        exit(0);
    }

    int n;
    in >> n;
    std::vector<int> data(n);
    for (int i = 0; i < n; i++) {
        int num = 0;
        in >> num;
        data.push_back(num);
    }

    try {

        BS::BitonicSorter bs("sort_kernel.cl");
        bs.Sort(data, BS::Dir::Increase);

        for (int i = 0; i < n; i++)
            std::cout<<data[i]<<" ";

    }
    catch (std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        exit(1);
    }

    return 0;
}


