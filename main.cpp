#include "BitonicSorter.h"
#include <random>

int main(int argc, char* argv[]) {

    std::istream& in = std::cin;
    /*std::ifstream in{"tests/test1.txt"};
    if (!in.is_open()) {
        std::cerr << "Cant open file!" << std::endl;
        exit(0);
    }*/

    int n;
    in >> n;
    std::vector<int> data(n);
    for (int i = 0; i < n; i++) {
        if (!in.good()) {
            std::cout<<"Problems with cin!"<<std::endl;
            exit(0);
        }
        in >> data[i];
    }

    try {

        BS::BitonicSorter bs;
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