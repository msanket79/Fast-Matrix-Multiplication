#ifndef KERNELS_HPP
#define KERNELS_HPP

#include<omp.h>
#include<iostream>
#include<cstdlib>
#include <immintrin.h>

// defs

#define convert(A,r,c) (A+(r*cols+c)*N/2)

//utils-----------------------------------------------------------------------------------------------------------------------
void InitVal(float* A, int m, int n, int val = -1) {
    if (val == -1)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = val;
        }
    }

}
void printMat(float* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

float MaxError(float* A, float* B, int m, int n) {
    float maxErr=INT_MIN;
    for(int i=0;i<m*n;i++){
        maxErr=std::max<float>(abs(B[i]-A[i]),maxErr);
    }
    return maxErr;
}
//--------------------------------------------------------------------------------------------------------------------------

// simple three nested for loops 
template<int M, int K, int N>
inline void matMulNaive(const float* A, const float* B, float* C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C[m * K + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}
// the m and k loops are reordered to increase the cache hits as previously B was accessed in column Major order
template<int M, int K, int N>
inline void matMulLoopReorder(const float* A, const float* B, float* C) {
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * K + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}
//parallelising the outer loop and distbuting work over all the cores of cpu
template<int M, int K, int N>
inline void matMulParallel(const float* A, const float* B, float* C) {
#pragma omp parallel for num_threads(8)
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * K + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}


//blocking  (tiling for L1 cache)
template<int M, int K, int N, int s>
inline void matMulTiling(const float* A, const float* B, float* C) {
#pragma omp parallel for 
    for (int mm = 0; mm < M; mm += s) {
        for (int nn = 0; nn < N; nn += s) {
            for (int kk = 0; kk < K; kk += s) {
                for (int m = 0; m < s; m++) {
                    for (int k = 0; k < s; k++) {
                        for (int n = 0; n < s; n++) {
                            C[(mm + m) * N + nn + n] += A[(mm + m) * K + kk + k] * B[(kk + k) * N + n];
                        }
                    }
                }
            }
        }
    }

}


//using recurssion to acheive tiling over all the caches
template<int cols, int threshold>
void matMulRec(float* A, float* B, float* C, int N) {
    if (N == threshold) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    C[i * cols + j] += A[i * cols + k] * B[k * cols + j];
                }
            }
        }
    }
    else {
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 0, 0), convert(B, 0, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 0, 0), convert(B, 1, 0), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 1, 0), convert(B, 0, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 1, 0), convert(B, 0, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait

#pragma omp task
        matMulRec<cols, threshold>(convert(A, 0, 1), convert(B, 1, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 0, 1), convert(B, 1, 1), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 1, 1), convert(B, 1, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulRec<cols, threshold>(convert(A, 1, 1), convert(B, 1, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait

    }
}

// using avx512 for matmul in the base case
template<int cols, int threshold>
void matMulVectorisation(float* A, float* B, float* C, int N) {
    if (N == threshold) {

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                __m512 avec = _mm512_set1_ps(A[i * cols + k]);
                for (int j = 0; j < N; j += 16) {
                    __m512 bvec = _mm512_load_ps(&B[k * cols + j]);
                    __m512 cvec = _mm512_load_ps(&C[i * cols + j]);
                    __m512 prod = _mm512_fmadd_ps(avec, bvec, cvec);
                    _mm512_store_ps(&C[i * cols + j], prod);
                }
            }
        }
    }
    else {
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 0, 0), convert(B, 0, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 0, 0), convert(B, 1, 0), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 1, 0), convert(B, 0, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 1, 0), convert(B, 0, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait

#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 0, 1), convert(B, 1, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 0, 1), convert(B, 1, 1), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 1, 1), convert(B, 1, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulVectorisation<cols, threshold>(convert(A, 1, 1), convert(B, 1, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait

    }
}

//now unrolling the base case inner loop 
template <int cols, int threshold>
void matMulVectorisationAndLoopUnrolling(float* A, float* B, float* C, int N)
{
    if (N == threshold)
    {
        int i, k;
        for (i = 0; i < 64; i++)
        {
            for (k = 0; k < 64; k++)
            {
                __m512 avec, bvec, bvec1, bvec2, bvec3, cvec, cvec1, cvec2, cvec3, prod, prod1, prod2, prod3;

                avec = _mm512_set1_ps(A[i * cols + k]);

                bvec = _mm512_load_ps(&B[k * cols]);
                cvec = _mm512_load_ps(&C[i * cols]);
                prod = _mm512_fmadd_ps(avec, bvec, cvec);
                _mm512_store_ps(&C[i * cols], prod);

                bvec1 = _mm512_load_ps(&B[k * cols + 16]);
                cvec1 = _mm512_load_ps(&C[i * cols + 16]);
                prod1 = _mm512_fmadd_ps(avec, bvec1, cvec1);
                _mm512_store_ps(&C[i * cols + 16], prod);

                bvec2 = _mm512_load_ps(&B[k * cols + 32]);
                cvec2 = _mm512_load_ps(&C[i * cols + 32]);
                prod2 = _mm512_fmadd_ps(avec, bvec2, cvec2);
                _mm512_store_ps(&C[i * cols + 32], prod);

                bvec3 = _mm512_load_ps(&B[k * cols + 48]);
                cvec3 = _mm512_load_ps(&C[i * cols + 48]);
                prod3 = _mm512_fmadd_ps(avec, bvec3, cvec3);
                _mm512_store_ps(&C[i * cols + 48], prod);
            }
        }
    }
    else
    {
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 0, 0), convert(B, 0, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 0, 0), convert(B, 1, 0), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 1, 0), convert(B, 0, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 1, 0), convert(B, 0, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait

#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 0, 1), convert(B, 1, 0), convert(C, 0, 0), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 0, 1), convert(B, 1, 1), convert(C, 0, 1), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 1, 1), convert(B, 1, 0), convert(C, 1, 0), N / 2);
#pragma omp task
        matMulVectorisationAndLoopUnrolling<cols, threshold>(convert(A, 1, 1), convert(B, 1, 1), convert(C, 1, 1), N / 2);
#pragma omp taskwait
    }
}

#endif