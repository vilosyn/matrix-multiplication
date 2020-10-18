#include <stdlib.h>
#define __CL_ENABLE_EXCEPTIONS
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CL/cl.hpp"
#include "omp.h"

void randomInit(std::vector<double> &data) {
  size_t j = 0;
  for (auto &i : data)
    i = rand() / (double)RAND_MAX;
}

void GenerateTestData(int &WIDTH_A, int &HEIGHT_A,
                      std::vector<double> &matrix_A,
                      std::vector<double> &matrix_B,
                      std::vector<double> &result_matrix,
                      std::vector<double> &result_matrix_host) {
  srand((size_t)time(NULL));
  std::cout << "Enter dimension bound: ";
  std::cin >> WIDTH_A;
  HEIGHT_A = WIDTH_A;
  size_t size_A = WIDTH_A * HEIGHT_A;
  matrix_A.resize(size_A);
  matrix_B.resize(size_A);
  randomInit(matrix_A);
  randomInit(matrix_B);
  result_matrix.resize(size_A);
  result_matrix_host.resize(size_A);
}

void PerformCalculationOnDevice(cl::Device device, int &WIDTH_A, int &HEIGHT_A,
                                std::vector<double> &matrix_A,
                                std::vector<double> &matrix_B,
                                std::vector<double> &result_matrix) {
  std::vector<cl::Device> contextDevices;
  contextDevices.push_back(device);
  cl::Context context(contextDevices);
  cl::CommandQueue queue(context, device);
  cl::Buffer cl_matrix_A, cl_matrix_B, cl_result_matrix;
  auto start_t = std::chrono::high_resolution_clock::now();
  cl_matrix_A = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                           WIDTH_A * HEIGHT_A * sizeof(double), &matrix_A[0]);
  cl_matrix_B = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                           WIDTH_A * HEIGHT_A * sizeof(double), &matrix_B[0]);
  cl_result_matrix =
      cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                 WIDTH_A * HEIGHT_A * sizeof(double), &result_matrix[0]);
  queue.finish();
  auto end_t = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_t - start_t;
  std::cout << "Context, queue, buffers " << diff.count() << " s\n";

  std::ifstream sourceFile("./matmul.cl");
  std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),
                         (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(
      1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
  cl::Program program = cl::Program(context, source);
  program.build(contextDevices);
  cl::Kernel kernel(program, "matrixVectorMul");
  int iArg = 0;
  kernel.setArg(iArg++, cl_result_matrix);
  kernel.setArg(iArg++, cl_matrix_A);
  kernel.setArg(iArg++, cl_matrix_B);
  kernel.setArg(iArg++, WIDTH_A);
  kernel.setArg(iArg++, HEIGHT_A);
  cl::NDRange global_work(int((WIDTH_A + 15) / 16) * 16,
                          int((WIDTH_A + 15) / 16) * 16);
  cl::NDRange local_work = cl::NDRange(16, 16);

  start_t = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work, local_work);
  queue.finish();
  end_t = std::chrono::high_resolution_clock::now();
  diff = end_t - start_t;
  std::cout << "enqueueNDRangeKernel    " << diff.count() << " s\n";
  start_t = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(cl_result_matrix, CL_TRUE, 0,
                          WIDTH_A * HEIGHT_A * sizeof(double),
                          &result_matrix[0]);
  queue.finish();
  end_t = std::chrono::high_resolution_clock::now();
  diff = end_t - start_t;
  std::cout << "enqueueReadBuffer       " << diff.count() << " s\n";
}

void PerformCalculationOnHost(int &WIDTH_A, int &HEIGHT_A,
                              std::vector<double> &matrix_A,
                              std::vector<double> &matrix_B,
                              std::vector<double> &result_matrix_host) {
#pragma omp parallel num_threads(16)
  {
#pragma omp for
    for (int i = 0; i < WIDTH_A; i++)
#pragma omp simd
      for (int j = 0; j < WIDTH_A; j++) {
        double tmp = 0;
        for (int k = 0; k < HEIGHT_A; k++)
          tmp += matrix_A[j * HEIGHT_A + k] * matrix_B[k * WIDTH_A + i];

        result_matrix_host[j * WIDTH_A + i] = tmp;
      }
  }
}

int main(int argc, char **argv) {
  int WIDTH_A, HEIGHT_A;
  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> result_matrix;
  std::vector<double> result_matrix_host;
  GenerateTestData(WIDTH_A, HEIGHT_A, matrix_A, matrix_B, result_matrix,
                   result_matrix_host);
  auto start_t = std::chrono::high_resolution_clock::now();
  PerformCalculationOnHost(WIDTH_A, HEIGHT_A, matrix_A, matrix_B,
                           result_matrix_host);
  auto end_t = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_t - start_t;
  std::cout << "Host:                   " << diff.count() << " s\n";
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::vector<cl::Device> devices;
  for (size_t iPlatform = 0; iPlatform < platforms.size(); iPlatform++) {
    auto start_t = std::chrono::high_resolution_clock::now();
    platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (size_t iDevice = 0; iDevice < devices.size(); iDevice++) {
      try {
        std::cout << "Platform: "
                  << platforms[iPlatform].getInfo<CL_PLATFORM_NAME>()
                  << std::endl;
        std::cout << "Device:   " << devices[iDevice].getInfo<CL_DEVICE_NAME>()
                  << std::endl;
        std::fill(result_matrix.begin(), result_matrix.end(), 0.);
        PerformCalculationOnDevice(devices[iDevice], WIDTH_A, HEIGHT_A,
                                   matrix_A, matrix_B, result_matrix);
      } catch (cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
      }
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_t - start_t;
    std::cout << "Device:                 " << diff.count() << " s\n";
    double max_deviation = 0;
    FILE *f, *f_host;
    f = fopen("device_result", "w");
    f_host = fopen("host_result", "w");
#pragma omp parallel for reduction(max : max_deviation) num_threads(16)
    for (int i = 0; i < WIDTH_A * HEIGHT_A; i++) {
      fprintf(f, "%.2f\n", result_matrix[i]);
      fprintf(f_host, "%.2f\n", result_matrix_host[i]);
      max_deviation = std::max(max_deviation,
                               fabs(result_matrix[i] - result_matrix_host[i]));
    }

    fclose(f);
    fclose(f_host);
    std::cout << "Max deviation = " << max_deviation << std::endl;
  }

  return 0;
}
