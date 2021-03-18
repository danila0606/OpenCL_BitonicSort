#include "BitonicSorter.h"
#include <set>
#include <random>

//for reading testfile names
#include <dirent.h>
#include <sys/stat.h>
std::vector<std::string> GetAllTestFileNames(const std::string& dirname);

//info for generating tests
struct TestGenInfo final {
    std::string name;
    int N, min_value, max_value;
    BS::Dir dir; // 0 - decrease
};
void TestGenerator(const std::vector<TestGenInfo>& files);


int main () {

    auto filenames = GetAllTestFileNames("tests/");

    try {
        for (const auto& elem : filenames) {

            std::ifstream istr(elem);

            if (!istr.is_open()) {
                std::cerr << "Cant open file! " << elem << std::endl;
                exit(0);
            }

            int n = 0;
            istr >> n;
            std::vector<int> data(n);
            for (int i = 0; i < n; ++i)
               istr>>data[i];

            istr.close();

            std::vector<int> copy_data(data);

            //bitonic sort - sorting
            BS::BitonicSorter bs("sort_kernel.cl");
            auto bitonic_time = bs.Sort(data, BS::Dir::Increase);

            //std::sort - sorting
            auto start_t_set = std::chrono::system_clock::now();
            std::sort(copy_data.begin(), copy_data.end()); // Increase
            auto finish_t_set = std::chrono::system_clock::now();

            if (data.size() != copy_data.size())
                throw std::invalid_argument("Array sizes don't match after sorting");

            //comparing results
            for (int i = 0; i < data.size(); ++i) {
                if (data[i] != copy_data[i])
                    throw std::invalid_argument("Arrays are not equal");
            }

            std::cout<<"-------- "<<elem<<" --------"<<std::endl;
            std::cout <<"Number of elements: "<< n <<std::endl;
            std::cout << "std::sort Time : "<< (finish_t_set - start_t_set).count()<<std::endl;
            std::cout << "Bitonic sort Time: "<<bitonic_time<<std::endl;
            std::cout<<"----------------------------"<<std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        exit(1);
    }


    /*TestGenInfo t1 = {"tests/test1.txt", 100, 0, 100, BS::Dir::Increase};
    TestGenInfo t2 = {"tests/test2.txt", 10000, -30000, 30000, BS::Dir::Increase};
    TestGenInfo t3 = {"tests/test3.txt", 20000, 0, 100000, BS::Dir::Decrease};
    TestGenerator({t1,t2,t3});*/

    return 0;
}

std::vector<std::string> GetAllTestFileNames(const std::string& dirname) {

    std::vector<std::string> filenames;

    DIR* dir = opendir(dirname.data());
    struct dirent* dirent_;
    while ((dirent_ = readdir(dir))!= nullptr) {

        char copystr[100];
        strcpy(copystr, dirname.data());
        strcat(copystr,dirent_-> d_name);

        struct stat buf{};
        lstat(copystr, &buf);

        if (S_ISREG(buf.st_mode))
            filenames.emplace_back(copystr);
    }
    closedir(dir);
    return  filenames;
}

void TestGenerator(const std::vector<TestGenInfo>& files) {

    for (const auto& file : files) {
        unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::default_random_engine eng(seed);
        std::uniform_int_distribution<int> distr(file.min_value, file.max_value);

        std::ofstream ostr(file.name);
        if (!ostr.is_open()) {
            std::cerr << "Cant open input file!" << std::endl;
            exit(0);
        }
        ostr<<file.N<<std::endl;
        for (int i = 0; i < file.N; ++i) {
            int key = distr(eng);
            ostr << key << " ";
        }

    }
};


