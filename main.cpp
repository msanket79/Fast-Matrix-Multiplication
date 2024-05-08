//In pervious method we were accessing C[i][j] in every iteration of k to it will be saved in cache and then will hit every time in cache
//But we can optimize it storing it in a varaible which is used again n again so it will saved in a register and will only take a single cycle for the value
//We are multiplying two matrices of 4096*4096
#include<iostream>
#include<omp.h>
#include<chrono>
#include "kernels.hpp"
#include<string>
#define A_val 2.0
#define B_val 3.0


int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cout << "Pass command line argument for which kernel to call\n"
            "-----------------------------------------------------------\n"
            "[.]0 for Mat Mul naive\n"
            "[.]1 for Mat Mul loop reorder\n"
            "[.]2 for Mat Mul parallel\n"
            "[.]3 for Mat Mul Tiled\n"
            "[.]4 for Mat Mul Recursive\n"
            "[.]5 for Mat Mul Vectorised\n"
            "[.]6 for Mat Mul Vectorised and unrolled\n";
        exit(0);

    }
    int kernelNo = std::stoi(argv[1]);

    float* A = (float*)malloc(sizeof(float) * 4096 * 4096);
    float* B = (float*)malloc(sizeof(float) * 4096 * 4096);
    float* C = (float*)malloc(sizeof(float) * 4096 * 4096);
    InitVal(A, 4096, 4096);
    InitVal(B, 4096, 4096);
    InitVal(C, 4096, 4096, 0);
    switch (kernelNo)
    {
    case 0:
    {
        auto start = std::chrono::high_resolution_clock::now();
        matMulNaive<4096, 4096, 4096>(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 1:
    {
        auto start = std::chrono::high_resolution_clock::now();
        matMulLoopReorder<4096, 4096, 4096>(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 2:
    {
        auto start = std::chrono::high_resolution_clock::now();
        matMulParallel<4096, 4096, 4096>(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 3:
    {
        auto start = std::chrono::high_resolution_clock::now();
        matMulTiling<4096, 4096, 4096, 64>(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 4:
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel 
        #pragma omp single
        matMulRec<4096, 64>(A, B, C, 4096);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 5:
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        #pragma omp single
        matMulVectorisation<4096, 64>(A, B, C, 4096);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;
    case 6:
    {
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel 
#pragma omp single
        matMulVectorisationAndLoopUnrolling<4096, 64>(A, B, C, 4096);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "Time taken by  is : " << duration.count()/1e9 << " sec" << std::endl;
    }
    break;

    default:
        std::cout << "Kernel doesn't exist\n";
        break;
    }
    return 0;
}