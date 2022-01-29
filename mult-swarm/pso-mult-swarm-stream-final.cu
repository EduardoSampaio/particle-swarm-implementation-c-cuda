#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
//#include "pso-mult-swarm-sequencial.h"

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

typedef struct Particle
{
    double *position_i;
    double *velocity_i;
    double *pos_best_i;
    double err_best_i = -1;
    double err_i = -1;
} Particle;

typedef struct Swarm
{
    Particle *swarm;
    double *pos_best_s;
    double *err_best_s;
} Swarm;

const double MaxValue = 1.7976931348623157E+308;
const int num_dimensions = 2;
const int num_particle = 4096;
const int THREAD_PER_BLOCK = 128;
const int BLOCKS = num_particle / THREAD_PER_BLOCK;
const int MAX_ITER = 30;

double h_initial[] = {5, 5};
double *initial, *BOUNDS, *numbersRand;
double h_BOUNDS[] = {-10, 10, -10, 10};
Swarm *swarms, *h_swarms, *out_swarm;
double *pos_best_g, *h_pos_best_g;
double *err_best_g, h_err_best_g;
Particle *swarm, *h_swarm;

__device__ double gpuRandomNumberUniform(curandState_t state)
{
    curand_init(0, 0, 0, &state);
    return curand_uniform(&state);
}

__device__ double sphere(Particle *x)
{
    double total = 0.0;
    for (int i = 0; i <= num_dimensions; i++)
    {
        total += pow(x->position_i[i], 2);
    }
    return total;
}

__global__ void gpuGenerateRand(double *numbers)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t state;
    curand_init(0, 0, i, &state);
    double number = curand_uniform(&state);
    numbers[i] = number;
}

// void initialize_gpu_memory(int num_swarms)
//{
//	h_pos_best_g = (double*)malloc(sizeof(double) * num_dimensions);
//	h_pos_best_g[0] = MaxValue;
//	h_pos_best_g[1] = MaxValue;
//	h_err_best_g = MaxValue;
//
//	h_swarm = (Particle*)malloc(sizeof(Particle) * num_particle);
//	h_swarms = (Swarm*)malloc(sizeof(Swarm) * num_swarms);
//	cudaMalloc(&pos_best_g, sizeof(double) * num_dimensions);
//	cudaMalloc(&err_best_g, sizeof(double));
//	cudaMalloc(&BOUNDS, sizeof(double) * num_dimensions * num_dimensions);
//	cudaMalloc(&initial, sizeof(double) * num_dimensions);
//	cudaMalloc(&numbersRand, sizeof(double) * num_dimensions * num_particle * num_swarms);
//
//	gpuGenerateRand << <BLOCKS, THREAD_PER_BLOCK >> > (numbersRand);
//
//	for (int k = 0; k < num_swarms; k++)
//	{
//		double h_err_best_s = MaxValue;
//		HANDLE_ERROR(cudaMalloc(&h_swarms[k].swarm, sizeof(Particle) * num_particle));
//		HANDLE_ERROR(cudaMalloc(&h_swarms[k].pos_best_s, sizeof(double) * num_dimensions));
//		HANDLE_ERROR(cudaMalloc(&h_swarms[k].err_best_s, sizeof(double)));
//
//		for (int i = 0; i < num_particle; i++)
//		{
//			HANDLE_ERROR(cudaMalloc(&h_swarm[i].position_i, sizeof(double) * num_dimensions));
//			HANDLE_ERROR(cudaMalloc(&h_swarm[i].velocity_i, sizeof(double) * num_dimensions));
//			HANDLE_ERROR(cudaMalloc(&h_swarm[i].pos_best_i, sizeof(double) * num_dimensions));
//		}
//
//		HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].err_best_s, &h_err_best_s, sizeof(double), cudaMemcpyHostToDevice));
//		HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].pos_best_s, &h_pos_best_g, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice));
//		HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].swarm, h_swarm, sizeof(Particle) * num_particle, cudaMemcpyHostToDevice));
//	}
//	cudaMemcpyAsync(BOUNDS, h_BOUNDS, sizeof(double) * num_dimensions * num_dimensions, cudaMemcpyHostToDevice);
//	cudaMemcpyAsync(initial, h_initial, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
//
// }

void initialize_gpu_memory(int num_swarms)
{
    h_pos_best_g = (double *)malloc(sizeof(double) * num_dimensions);
    h_pos_best_g[0] = MaxValue;
    h_pos_best_g[1] = MaxValue;
    h_err_best_g = MaxValue;

    cudaMallocHost(&h_swarm, sizeof(Particle) * num_particle);
    cudaMallocHost(&h_swarms, sizeof(Swarm) * num_swarms);
    cudaMalloc(&pos_best_g, sizeof(double) * num_dimensions);
    cudaMalloc(&err_best_g, sizeof(double));
    cudaMalloc(&BOUNDS, sizeof(double) * num_dimensions * num_dimensions);
    cudaMalloc(&initial, sizeof(double) * num_dimensions);
    cudaMalloc(&numbersRand, sizeof(double) * num_dimensions * num_particle * num_swarms);

    gpuGenerateRand<<<BLOCKS, THREAD_PER_BLOCK>>>(numbersRand);

    for (int k = 0; k < num_swarms; k++)
    {
        double h_err_best_s = MaxValue;
        HANDLE_ERROR(cudaMalloc(&h_swarms[k].swarm, sizeof(Particle) * num_particle));
        HANDLE_ERROR(cudaMalloc(&h_swarms[k].pos_best_s, sizeof(double) * num_dimensions));
        HANDLE_ERROR(cudaMalloc(&h_swarms[k].err_best_s, sizeof(double)));

        for (int i = 0; i < num_particle; i++)
        {
            HANDLE_ERROR(cudaMalloc(&h_swarm[i].position_i, sizeof(double) * num_dimensions));
            HANDLE_ERROR(cudaMalloc(&h_swarm[i].velocity_i, sizeof(double) * num_dimensions));
            HANDLE_ERROR(cudaMalloc(&h_swarm[i].pos_best_i, sizeof(double) * num_dimensions));
        }

        HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].err_best_s, &h_err_best_s, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].pos_best_s, &h_pos_best_g, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyAsync(h_swarms[k].swarm, h_swarm, sizeof(Particle) * num_particle, cudaMemcpyHostToDevice));
    }
    cudaMemcpyAsync(BOUNDS, h_BOUNDS, sizeof(double) * num_dimensions * num_dimensions, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(initial, h_initial, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
}

__global__ void init_particle(Particle *swarm, double *x0, double *numbers, int num_swarms)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (int j = 0; j < num_dimensions; j++)
    {
        swarm[i].position_i[j] = x0[j];
        swarm[i].velocity_i[j] = numbers[i + j];
    }
}

__global__ void pso(Particle *swarm, double *pos_best_s, double *err_best_s, double *bounds, double *numbers)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    swarm[i].err_i = sphere(&swarm[i]);

    if (swarm[i].err_i < swarm[i].err_best_i)
    {
        swarm[i].pos_best_i = swarm[i].position_i;
        swarm[i].err_best_i = swarm[i].err_i;
    }

    if (swarm[i].err_i < *err_best_s || *err_best_s == -1)
    {
        pos_best_s[0] = swarm[i].position_i[0];
        pos_best_s[1] = swarm[i].position_i[1];
        *err_best_s = swarm[i].err_i;
    }

    double w = 0.9;
    int c1 = 1;
    int c2 = 2;

    for (int j = 0; j < num_dimensions; j++)
    {
        swarm[i].position_i[j] = swarm[i].position_i[j] + swarm[i].velocity_i[j];
        if (swarm[i].position_i[j] > bounds[j * num_dimensions + 1])
        {
            swarm[i].position_i[j] = bounds[j * num_dimensions + 1];
        }

        if (swarm[i].position_i[j] < bounds[j * num_dimensions + 0])
        {
            swarm[i].position_i[j] = bounds[j * num_dimensions + 0];
        }

        double r1 = numbers[i];
        double r2 = numbers[i + j];
        double vel_cognitive = c1 * r1 * (swarm[i].pos_best_i[j] - swarm[i].position_i[j]);
        double vel_social = c2 * r2 * (pos_best_s[j] - swarm[i].position_i[j]);
        swarm[i].velocity_i[j] = w * swarm[i].velocity_i[j] + vel_cognitive + vel_social;
    }
}

void pso_execute_stream(int num_swarms, int repeat)
{
    float gtime = 0.0;
    for (int t = 0; t < repeat; t++)
    {

        cudaDeviceReset();

        float time = 0.0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        initialize_gpu_memory(num_swarms);
        cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_swarms);
        double h_pos_best_s[2];
        double h_err_best_s;
        int k = 0;
        for (int i = 0.; i < num_swarms; i++)
        {
            init_particle<<<BLOCKS, THREAD_PER_BLOCK>>>(h_swarms[i].swarm, initial, numbersRand, num_swarms);
            HANDLE_ERROR(cudaStreamCreate(&stream[i]));
        }

        cudaDeviceSynchronize();
        while (k < MAX_ITER)
        {
            for (int i = 0.; i < num_swarms; i++)
            {
                pso<<<BLOCKS, THREAD_PER_BLOCK, 0, stream[i]>>>(h_swarms[i].swarm, h_swarms[i].pos_best_s, h_swarms[i].err_best_s, BOUNDS, numbersRand);
            }
            k++;
        }

        cudaDeviceSynchronize();
        for (int i = 0.; i < num_swarms; i++)
        {
            HANDLE_ERROR(cudaMemcpyAsync(h_pos_best_s, h_swarms[i].pos_best_s, sizeof(double) * num_dimensions, cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpyAsync(&h_err_best_s, h_swarms[i].err_best_s, sizeof(double), cudaMemcpyDeviceToHost));
            if (h_pos_best_s[0] < h_pos_best_g[0] && h_pos_best_s[1] < h_pos_best_g[1])
            {
                h_pos_best_g[0] = h_pos_best_s[0];
                h_pos_best_g[1] = h_pos_best_s[1];
                h_err_best_g = h_err_best_s;
            }
            printf("Solution Swarm:%d [x:%.20f, y:% .20f] error: % .20f\n", i, h_pos_best_s[0], h_pos_best_s[1], h_err_best_s);
        }

        printf("Final Solution: [x:%.20f, y:% .20f] error: % .20f\n", h_pos_best_g[0], h_pos_best_g[1], h_err_best_g);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);

        gtime += time;

        for (int i = 0; i < num_swarms; i++)
        {
            HANDLE_ERROR(cudaStreamDestroy(stream[i]));
        }

        cudaError_t err = cudaGetLastError();
        HANDLE_ERROR(err);
    }

    printf("Result Mult-Stream Swarms: %d Time: %3.2f ms.\n", num_swarms, gtime / repeat);
}

int main()
{

    cudaProfilerStart();
    pso_execute_stream(16, 1);
    cudaProfilerStop();
    return 0;
}
