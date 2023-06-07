# PCA-Matrix-summation-with-a-2D-grid-and-2D-blocks.-Adapt-it-to-integer-matrix-addition.-

## Aim:

To perform GPU based vector summation and explore the differences with different block values. 

## Procedure:
• Define the size of the matrix by setting the values of nx and ny, and calculate the total 
number of elements, nxy.

• Allocate memory on both the host and device for the input and output matrices using the 
appropriate data types.

• Initialize the input matrices with data on the host side and transfer them to the device 
memory using the cudaMemcpy function.

• Define the block and grid dimensions for the kernel. The block dimensions should be a 2D 
array with each element representing the number of threads in each dimension. The grid 
dimensions should be calculated using the formula (ceil(nx/block.x), ceil(ny/block.y)).

• Launch the kernel with the input and output matrices as arguments using the <<<grid, 
block>>> notation. Wait for the kernel to finish executing using the cudaDeviceSynchronize 
function.

• Transfer the output matrix from device to host memory using the cudaMemcpy function.

• Check the correctness of the output matrix by comparing it to the expected result on the 
host side.

• Free the memory on both the host and device using the appropriate functions. Reset the 
device using the cudaDeviceReset function.

##code:
sumMatrixOnGPU-2D-grid-2D-block.cu:

#include "common.h"

#include <cuda_runtime.h>

#include <stdio.h>

void initialData(float *ip, const int size)

{

 int i;
 
 for(i = 0; i < size; i++)
 {
 
 ip[i] = (float)(rand() & 0xFF) / 10.0f;
 
 }
 
 return;
 
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
 float *ia = A;
 
 float *ib = B;
 
 float *ic = C;
 
 for (int iy = 0; iy < ny; iy++)
 
 {
 for (int ix = 0; ix < nx; ix++)
 
 {
 
 ic[ix] = ia[ix] + ib[ix];
 
 }
 
 ia += nx;
 
 ib += nx;
 
 ic += nx;
 
 }
 
 return;
 
}

void checkResult(float *hostRef, float *gpuRef, const int N)

{

 double epsilon = 1.0E-8;
 
 bool match = 1;
 
 for (int i = 0; i < N; i++)
 
 {
 
 if (abs(hostRef[i] - gpuRef[i]) > epsilon)
 
 {
 
 match = 0;
 
 printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
 
 break;
 
 }
 
 }
 
 if (match)
 
 printf("Arrays match.\n\n");
 
 else
 
 printf("Arrays do not match.\n\n");
 
}

// grid 2D block 2D

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)

{

 unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
 
 unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
 
 unsigned int idx = iy * nx + ix;
 
 if (ix < nx && iy < ny)
 
 MatC[idx] = MatA[idx] + MatB[idx];
 
}

int main(int argc, char **argv)

{

 printf("%s Starting...\n", argv[0]);
 
 // set up device
 int dev = 0;
 
 cudaDeviceProp deviceProp;
 
 CHECK(cudaGetDeviceProperties(&deviceProp, dev));
 
 printf("Using Device %d: %s\n", dev, deviceProp.name);
 
 CHECK(cudaSetDevice(dev));
 
 // set up data size of matrix
 
 int nx = 1 << 14;
 
 int ny = 1 << 14;
 
 int nxy = nx * ny;
 
 int nBytes = nxy * sizeof(float);
 
 printf("Matrix size: nx %d ny %d\n", nx, ny);
 
 // malloc host memory
 
 float *h_A, *h_B, *hostRef, *gpuRef;
 
 h_A = (float *)malloc(nBytes);
 
 h_B = (float *)malloc(nBytes);
 
 hostRef = (float *)malloc(nBytes);
 
 gpuRef = (float *)malloc(nBytes);
 
 // initialize data at host side
 
 double iStart = seconds();
 
 initialData(h_A, nxy);
 
 initialData(h_B, nxy);
 
 double iElaps = seconds() - iStart;
 
 printf("Matrix initialization elapsed %f sec\n", iElaps);
 
 memset(hostRef, 0, nBytes);
 
 memset(gpuRef, 0, nBytes);
 
 // add matrix at host side for result checks
 iStart = seconds();
 
   sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
   
   iElaps = seconds() - iStart;
   
   printf("sumMatrixOnHost elapsed %f sec\n", iElaps);
   
   // malloc device global memory
   
   
   float *d_MatA, *d_MatB, *d_MatC;
 CHECK(cudaMalloc((void **)&d_MatA, nBytes));
 
 CHECK(cudaMalloc((void **)&d_MatB, nBytes));
 
 CHECK(cudaMalloc((void **)&d_MatC, nBytes));
 
 // transfer data from host to device
 
 CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
 
 CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));
 
 // invoke kernel at host side
 
 int dimx = 32;
 
 int dimy = 32;
 
 dim3 block(dimx, dimy);
 
 dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
 
 iStart = seconds();
 
 sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
 
 CHECK(cudaDeviceSynchronize());
 
 iElaps = seconds() - iStart;
 
 printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x,
 block.y, iElaps);
 // check kernel error
 
 CHECK(cudaGetLastError());
 
 // copy kernel result back to host side
 
 CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
 
 // check device results
 
 checkResult(hostRef, gpuRef, nxy);
 
 // free device global memory
 
 CHECK(cudaFree(d_MatA));
 
 CHECK(cudaFree(d_MatB));
 
 CHECK(cudaFree(d_MatC));
 
 // free host memory
 
 free(h_A);
 
 free(h_B);
 
 free(hostRef);
 
 free(gpuRef);
 
 // reset device
 
 CHECK(cudaDeviceReset());
 
 return (0);
 
}

## Output:

![image](https://github.com/newuserno2/PCA-Matrix-summation-with-a-2D-grid-and-2D-blocks.-Adapt-it-to-integer-matrix-addition.-/assets/133124928/1ad45c3c-46b3-4726-883e-85b753124fd9)

## Result:

Thus, the Matrix Summation with a 2D grid and 2D blocks. And adapt it to Integer Matrix Addition 
has been successfully performed.
