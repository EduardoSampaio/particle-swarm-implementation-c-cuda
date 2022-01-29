#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

static void HandleError(cudaError_t err,
	const char* file,
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


const double MaxValue = 1.7976931348623157E+308;
const int num_dimensions = 2;
const int num_particle = 64;
const int THREAD_PER_BLOCK = 32;
const int BLOCKS = num_particle / THREAD_PER_BLOCK;
const int MAX_ITER = 10;

double h_initial[] = { 5, 5 };
double* initial, * BOUNDS, * numbersRand;
double h_BOUNDS[] = { -10, 10, -10, 10 };
double* pos_best_g, * h_pos_best_g;
double* err_best_g, h_err_best_g;


__device__ double gpuRandomNumberUniform(curandState_t state)
{
	curand_init(0, 0, 0, &state);
	return curand_uniform(&state);
}

__device__ double sphere(double* position_i)
{
	double total = 0.0;
	for (int i = 0; i <= num_dimensions; i++)
	{
		total += pow(position_i[i], 2);
	}
	return total;
}

__global__ void gpuGenerateRand(double* numbers)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t state;
	curand_init(0, 0, i, &state);
	double number = curand_uniform(&state);
	numbers[i] = number;
}


void initialize_gpu_memory(int num_swarms)
{
	h_pos_best_g = (double*)malloc(sizeof(double) * num_dimensions);
	h_pos_best_g[0] = MaxValue;
	h_pos_best_g[1] = MaxValue;
	h_err_best_g = MaxValue;

	cudaMalloc(&pos_best_g, sizeof(double) * num_dimensions);
	cudaMalloc(&err_best_g, sizeof(double));
	cudaMalloc(&BOUNDS, sizeof(double) * num_dimensions * num_dimensions);
	cudaMalloc(&initial, sizeof(double) * num_dimensions);
	cudaMalloc(&numbersRand, sizeof(double) * num_dimensions * num_particle * num_swarms);

	gpuGenerateRand << <BLOCKS, THREAD_PER_BLOCK >> > (numbersRand);
	cudaMemcpy(BOUNDS, h_BOUNDS, sizeof(double) * num_dimensions * num_dimensions, cudaMemcpyHostToDevice);
	cudaMemcpy(initial, h_initial, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
}



__global__ void init_particle(double* position_i, double* velocity_i, double* x0, double* numbers)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	for (int j = 0; j < num_dimensions; j++)
	{
		position_i[i + j] = x0[j];
		velocity_i[i + j] = numbers[i + j];
	}

	//printf("Index: %d [x:%.20f, y:% .20f] \n", i, position_i[i], position_i[i + 1]);
}

__global__ void pso(double* position_i, double* velocity_i, double* pos_best_i, double err_best_i, double err_i, double* pos_best_s, double* err_best_s, double* bounds, double* numbers)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	double total = 0.0;
	for (int k = 0; k < num_dimensions; k++)
	{
		total += pow(position_i[i + k], 2);
	}
	err_i = total;


	for (int j = 0; j < num_dimensions; j++) {
		if (err_i < err_best_i)
		{
			pos_best_i[i + j] = position_i[i + j];
			err_best_i = err_i;
		}

		if (err_i < *err_best_s)
		{
			pos_best_s[0] = position_i[i + j];
			pos_best_s[1] = position_i[i + j];
			*err_best_s = err_i;
		}
	}

	//printf("index: %d  [x:%.20f, y:% .20f] err_i: %f err_best_i %f\n", i, pos_best_s[0], pos_best_s[1], err_i, err_best_i);


	double w = 0.9;
	int c1 = 1;
	int c2 = 2;

	for (int j = 0; j < num_dimensions; j++)
	{
		position_i[i + j] = position_i[i + j] + velocity_i[i + j];
		if (position_i[i + j] > bounds[j * num_dimensions + 1])
		{
			position_i[i + j] = bounds[j * num_dimensions + 1];
		}

		if (position_i[i + j] < bounds[j * num_dimensions + 0])
		{
			position_i[i + j] = bounds[j * num_dimensions + 0];
		}

		double r1 = numbers[i];
		double r2 = numbers[i + j];
		double vel_cognitive = c1 * r1 * (pos_best_i[i + j] - position_i[i + j]);
		double vel_social = c2 * r2 * (pos_best_s[i + j] - position_i[i + j]);
		velocity_i[i + j] = w * velocity_i[i + j] + vel_cognitive + vel_social;
	}

	//printf("Solution: [x:%.20f, y:% .20f] error: % .20f\n", pos_best_s[0], pos_best_s[1], *err_best_s);
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
		cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_swarms);

		for (int i = 0.; i < num_swarms; i++)
		{
			HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		}

		double h_pos_best_s[2];
		double h_err_best_s;
		int k = 0;
	
		double* d_pos_best_s, * d_err_best_s, * pos_best_s, * err_best_s;
		double* d_position_i, * d_velocity_i, * d_pos_best_i, d_err_best_i, d_err_i;
		double* position_i, * velocity_i, * pos_best_i, err_best_i, err_i;
	

		HANDLE_ERROR(cudaMallocHost(&position_i, sizeof(double) * (num_particle * num_dimensions), cudaHostAllocDefault));
		HANDLE_ERROR(cudaMallocHost(&velocity_i, sizeof(double) * (num_particle * num_dimensions), cudaHostAllocDefault));
		HANDLE_ERROR(cudaMallocHost(&pos_best_i, sizeof(double) * (num_particle * num_dimensions), cudaHostAllocDefault));

		HANDLE_ERROR(cudaMalloc(&d_position_i, sizeof(double) * (num_particle * num_dimensions)));
		HANDLE_ERROR(cudaMalloc(&d_velocity_i, sizeof(double) * (num_particle * num_dimensions)));
		HANDLE_ERROR(cudaMalloc(&d_pos_best_i, sizeof(double) * (num_particle * num_dimensions)));

		HANDLE_ERROR(cudaMallocHost(&pos_best_s, sizeof(double) * num_dimensions, cudaHostAllocDefault));
		HANDLE_ERROR(cudaMallocHost(&err_best_s, sizeof(double), cudaHostAllocDefault));
		HANDLE_ERROR(cudaMalloc(&d_pos_best_s, sizeof(double) * num_dimensions));
		HANDLE_ERROR(cudaMalloc(&d_err_best_s, sizeof(double)));

		d_err_best_i = MaxValue;
		d_err_i = MaxValue;
		*err_best_s = MaxValue;

		HANDLE_ERROR(cudaMemcpy(d_position_i, position_i, sizeof(double) * (num_particle * num_dimensions), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_velocity_i, velocity_i, sizeof(double) * (num_particle * num_dimensions), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_pos_best_i, pos_best_i, sizeof(double) * (num_particle * num_dimensions), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(d_pos_best_s, pos_best_s, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_err_best_s, err_best_s, sizeof(double), cudaMemcpyHostToDevice));


		while (k < MAX_ITER)
		{
			for (int j = 0.; j < num_swarms; j++)
			{
				if (k == 0) {

					init_particle << <BLOCKS, THREAD_PER_BLOCK>> > (d_position_i, d_velocity_i, initial, numbersRand);
				}
				pso << <BLOCKS, THREAD_PER_BLOCK>> > (d_position_i, d_velocity_i, d_pos_best_i, d_err_best_i, d_err_i, d_pos_best_s, d_err_best_s, BOUNDS, numbersRand);

				HANDLE_ERROR(cudaMemcpy(h_pos_best_s, d_pos_best_s, sizeof(double) * num_dimensions, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemcpy(&h_err_best_s, d_err_best_s, sizeof(double), cudaMemcpyDeviceToHost));
				if (h_pos_best_s[0] < h_pos_best_g[0] && h_pos_best_s[1] < h_pos_best_g[1])
				{
					h_pos_best_g[0] = h_pos_best_s[0];
					h_pos_best_g[1] = h_pos_best_s[1];
					h_err_best_g = h_err_best_s;
				}
				printf("Solution Swarm:%d [x:%.20f, y:% .20f] error: % .20f\n", j, h_pos_best_s[0], h_pos_best_s[1], h_err_best_s);
			}
			k++;
		}

		cudaDeviceSynchronize();
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
