/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <thread>
using namespace std;

GPUTransformer::GPUTransformer (float* point_host_, int size_, int* ring_, int* sector_, int* height_, int max_length_, int num_ring_, int num_sector_, int num_height_, int enough_large_) {
  point_host = point_host_;
  h_max_length = max_length_;
  h_num_ring = num_ring_;
  h_num_height = num_height_;
  h_num_sector = num_sector_;
  enough_large = enough_large_;
  // h_ring = ring_;
  // h_sector = sector_;
  size = size_* 3 * sizeof(float);
  d_size = size_;
  int grid_size = num_ring_ * num_sector_ * num_height_ * enough_large * sizeof(int);
  d_grid_size = num_ring_ * num_sector_ * num_height_ ;
  // printf("num_height_ %d \n", num_height_);
  cudaMalloc((void**) &ring, d_size * sizeof(int));
  cudaMalloc((void**) &sector, d_size * sizeof(int));
  cudaMalloc((void**) &height, d_size * sizeof(int));

  // auto t1 = std::chrono::high_resolution_clock::now();
  cudaMalloc((void**) &point_device, size);

  // printf("err0 %s\n",cudaGetErrorString(err));

  // auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "memcpy took "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
  //           << " milliseconds\n";

  cudaMemcpy(point_device, point_host, size, cudaMemcpyHostToDevice);
  cudaMemcpy(sector, sector_, d_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(height, height_, d_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ring, ring_, d_size * sizeof(int), cudaMemcpyHostToDevice);
}

void GPUTransformer::transform() {
  dim3 blockSize(256);
  dim3 gridSize((d_size + blockSize.x - 1) / blockSize.x);
  point2gridmap<<<gridSize, blockSize>>>(point_device, ring, sector, height, d_size, h_max_length, h_num_ring, h_num_sector, h_num_height);
  cudaDeviceSynchronize();
}

void GPUTransformer::retreive(float* point_transformed) {
  int pt_count = 0;
  int index = 0;
  int ring_h[d_size] = {0};
  int sector_h[d_size] = {0};
  int height_h[d_size] = {0};
  int counter[d_grid_size] = {0};

  cudaMemcpy(ring_h, ring, d_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sector_h, sector, d_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(height_h, height, d_size * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < d_size; i++)
  {
    if(counter[sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring] < enough_large) //point_host[i + 2*d_size] >= (point_transformed[3*(ring_h[i] * h_num_sector + sector_h[i] + pt_count * h_num_sector * h_num_ring) + 2]) &&
    {
      pt_count = counter[sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring];
      // grid[sector_h[i] + ring_h[i] * h_num_sector + height[i] * h_num_sector * h_num_ring] = i;
      // mask[sector_h[i] + ring_h[i] * h_num_sector + height[i] * h_num_sector * h_num_ring] = 1;
      point_transformed[3*(sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring + pt_count * h_num_sector * h_num_ring * h_num_height) + 0] = point_host[i];
      point_transformed[3*(sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring + pt_count * h_num_sector * h_num_ring * h_num_height) + 1] = point_host[i + d_size];
      point_transformed[3*(sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring + pt_count * h_num_sector * h_num_ring * h_num_height) + 2] = point_host[i + 2*d_size];

      // for (int j = 0; j < pt_count+1; j++){
      //   point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + j * h_num_sector * h_num_ring) + 3] = x_ave[ring_h[i] * h_num_sector + sector_h[i]];
      //   point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + j * h_num_sector * h_num_ring) + 4] = y_ave[ring_h[i] * h_num_sector + sector_h[i]];
      //   point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + j * h_num_sector * h_num_ring) + 5] = z_ave[ring_h[i] * h_num_sector + sector_h[i]];        
      // }
      // point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + pt_count * h_num_sector * h_num_ring) + 3] = x_ave[ring_h[i] * h_num_sector + sector_h[i]];
      // point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + pt_count * h_num_sector * h_num_ring) + 4] = y_ave[ring_h[i] * h_num_sector + sector_h[i]];
      // point_transformed[6*(ring_h[i] * h_num_sector + sector_h[i] + pt_count * h_num_sector * h_num_ring) + 5] = z_ave[ring_h[i] * h_num_sector + sector_h[i]];

      // std::cout << "height " << height[i] << std::endl;
      counter[sector_h[i] + ring_h[i] * h_num_sector + height_h[i] * h_num_sector * h_num_ring] ++;
    }
  }

  cudaFree(point_device);
  cudaFree(height);
  cudaFree(sector);
  cudaFree(ring);
}

void GPUTransformer::retreive_to (int* array_host_, int length_) {

}

GPUTransformer::~GPUTransformer() {
  cudaFree(point_device);
  cudaFree(height);
  // cudaFree(d_max_length);
  // cudaFree(d_num_ring);
  cudaFree(sector);
  cudaFree(ring);
}
