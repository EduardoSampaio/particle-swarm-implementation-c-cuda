#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

typedef struct Particle
{
    double *position_i;
    double *velocity_i;
    double *pos_best_i;
    double err_best_i = -1;
    double err_i = -1;
} Particle;

double RNG_RANDOM()
{
    return rand() / (double)RAND_MAX;
}

double RNG_UNIFORM(int a, int b)
{
    return (a + (b - a) * RNG_RANDOM());
}

__device__ double gpuRandomNumberUniform(curandState_t state)
{
    curand_init(0, 0, 0, &state);
    return curand_uniform(&state);
}

const int num_dimensions = 2;

double BOUNDS_SPHERE[] = {-10, 10, -10, 10};
double BOUNDS_ROSENBROCK[] = {-2048, 2048, -2048, 2048};
double BOUNDS_RASTRINGIN[] = {-5.12, 5.12, -5.12, 5.12};
double BOUNDS_SCHWEFEL[] = {-500, 500, -500, 500};

__device__ double sphere(Particle *x)
{
    double total = 0.0;
    for (int i = 0; i <= num_dimensions; i++)
    {
        total += pow(x->position_i[i], 2);
    }
    return total;
}

__device__ double rosenbrock(Particle *x)
{
    double total = 0.0;
    for (int i = 0; i <= num_dimensions; i++)
    {
        total += 100 * (pow(x->position_i[i], 2) - pow(x->position_i[i + 1], 2)) + pow(1 - x->position_i[i], 2);
    }
    return total;
}

__device__ double rastrigin(Particle *x)
{
    double total = 0.0;
    for (int i = 0; i <= num_dimensions; i++)
    {
        total += (pow(x->position_i[i], 2) - (10 * cos(2 * 3.14159265359 * x->position_i[i])) + 10);
    }
    return total;
}

__device__ double schwefel(Particle *x)
{
    double total = 0.0;
    for (int i = 0; i <= num_dimensions; i++)
    {
        total += (x->position_i[i] * sin(sqrt(fabs(x->position_i[i]))));
    }
    return -total;
}

__global__ void update_gbest(Particle *swarm, double *pos_best_g, double *err_best_g)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (swarm[i].err_i < *err_best_g || *err_best_g == -1)
    {
        pos_best_g[0] = swarm[i].position_i[0];
        pos_best_g[1] = swarm[i].position_i[1];
        *err_best_g = swarm[i].err_i;
    }
}

__global__ void update_position_velocity(Particle *swarm, double *bounds, double *pos_best_g, double *numbers)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
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
        double vel_social = c2 * r2 * (pos_best_g[j] - swarm[i].position_i[j]);
        swarm[i].velocity_i[j] = w * swarm[i].velocity_i[j] + vel_cognitive + vel_social;
    }
}

__global__ void calculate_fitness(Particle *swarm)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    double err = sphere(&swarm[i]);
    swarm[i].err_i = err;
}

__global__ void evaluate_update_pbest(Particle *swarm)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (swarm[i].err_i < swarm[i].err_best_i || swarm[i].err_i == -1)
    {
        swarm[i].pos_best_i = swarm[i].position_i;
        swarm[i].err_best_i = swarm[i].err_i;
    }
}

__global__ void init_particle(Particle *swarm, double *x0, double *numbers)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < num_dimensions; j++)
    {
        swarm[i].position_i[j] = x0[j];
        swarm[i].velocity_i[j] = numbers[i + j];
    }
}

__global__ void gpuGenerateRand(double *numbers)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t state;
    curand_init(0, 0, i, &state);
    double number = curand_uniform(&state);
    numbers[i] = number;
}

int main()
{
    double h_initial[] = {5, 5};
    double *initial;
    const int num_particle = 4096;
    double h_BOUNDS[] = {-10, 10, -10, 10};
    double *BOUNDS;
    double *numbersRand;

    double *pos_best_g, *h_pos_best_g;
    double *err_best_g, *h_err_best_g;
    Particle *swarm, *h_swarm;
    const int THREAD_PER_BLOCK = 128;
    const int BLOCKS = num_particle / THREAD_PER_BLOCK;
    const int MAX_ITER = 30;

    float time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    h_pos_best_g = (double *)malloc(sizeof(double) * num_dimensions);
    h_err_best_g = (double *)malloc(sizeof(double));
    h_swarm = (Particle *)malloc(sizeof(Particle) * num_particle);
    cudaMalloc(&pos_best_g, sizeof(double) * num_dimensions);
    cudaMalloc(&err_best_g, sizeof(double));
    cudaMalloc(&swarm, sizeof(Particle) * num_particle);
    cudaMalloc(&BOUNDS, sizeof(double) * num_dimensions * num_dimensions);
    cudaMalloc(&initial, sizeof(double) * num_dimensions);
    cudaMalloc(&numbersRand, sizeof(double) * num_dimensions * num_particle);

    *h_err_best_g = -1;
    cudaMemcpy(pos_best_g, h_pos_best_g, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);
    cudaMemcpy(err_best_g, h_err_best_g, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(BOUNDS, h_BOUNDS, sizeof(double) * num_dimensions * num_dimensions, cudaMemcpyHostToDevice);
    cudaMemcpy(initial, h_initial, sizeof(double) * num_dimensions, cudaMemcpyHostToDevice);

    for (int i = 0; i < num_particle; i++)
    {
        cudaMalloc(&h_swarm[i].position_i, sizeof(double) * num_dimensions);
        cudaMalloc(&h_swarm[i].velocity_i, sizeof(double) * num_dimensions);
        cudaMalloc(&h_swarm[i].pos_best_i, sizeof(double) * num_dimensions);
    }

    cudaMemcpy(swarm, h_swarm, sizeof(Particle) * num_particle, cudaMemcpyHostToDevice);
    gpuGenerateRand<<<BLOCKS, THREAD_PER_BLOCK>>>(numbersRand);
    init_particle<<<BLOCKS, THREAD_PER_BLOCK>>>(swarm, initial, numbersRand);

    for (int i = 0; i < MAX_ITER; i++)
    {
        calculate_fitness<<<BLOCKS, THREAD_PER_BLOCK>>>(swarm);
        evaluate_update_pbest<<<BLOCKS, THREAD_PER_BLOCK>>>(swarm);
        update_gbest<<<BLOCKS, THREAD_PER_BLOCK>>>(swarm, pos_best_g, err_best_g);
        update_position_velocity<<<BLOCKS, THREAD_PER_BLOCK>>>(swarm, BOUNDS, pos_best_g, numbersRand);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(h_pos_best_g, pos_best_g, sizeof(double) * num_dimensions, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_err_best_g, err_best_g, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Time elapsed on CPU: %3.2f ms.\n", time);
    printf("Final Solution: [x:%.20f, y:% .20f] error: % .20f\n", h_pos_best_g[0], h_pos_best_g[1], *h_err_best_g);
    cudaFree(swarm);
    cudaFree(pos_best_g);
    cudaFree(swarm);
    cudaFree(BOUNDS);
}
