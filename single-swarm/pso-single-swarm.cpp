#include "benchmark.h"

#include <iostream>
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std;

double RNG_RANDOM()
{
    return rand() / (double)RAND_MAX;
}

double RNG_UNIFORM(int a, int b)
{
    return (a + (b - a) * RNG_RANDOM());
}
typedef struct Particle
{
    double *position_i;
    double *velocity_i;
    double *pos_best_i;
    double err_best_i = -1;
    double err_i = -1;
} Particle;

void alloc_memory(Particle *particle, int num_dimensions)
{
    particle->position_i = (double *)malloc(sizeof(double) * num_dimensions);
    particle->velocity_i = (double *)malloc(sizeof(double) * num_dimensions);
    particle->pos_best_i = (double *)malloc(sizeof(double) * num_dimensions);
}

void init_pso(Particle *particle, int num_dimensions, double *x0)
{
    for (int i = 0; i < num_dimensions; i++)
    {
        particle->velocity_i[i] = RNG_UNIFORM(-1, 1);
        particle->position_i[i] = x0[i];
        particle->pos_best_i[i] = 0.0;
    }
}

void evaluate(Particle *particle, Function function, int num_dimensions)
{
    particle->err_i = function(particle->position_i, num_dimensions);
    if (particle->err_i < particle->err_best_i || particle->err_i == -1)
    {
        particle->pos_best_i = particle->position_i;
        particle->err_best_i = particle->err_i;
    }
}

void update_velocity(Particle *particle, double *pos_best_g, int num_dimensions)
{
    double w = 0.5;
    int c1 = 1;
    int c2 = 2;
    for (int i = 0; i < num_dimensions; i++)
    {
        double r1 = RNG_RANDOM();
        double r2 = RNG_RANDOM();
        double vel_cognitive = c1 * r1 * (particle->pos_best_i[i] - particle->position_i[i]);
        double vel_social = c2 * r2 * (pos_best_g[i] - particle->position_i[i]);
        particle->velocity_i[i] = w * particle->velocity_i[i] + vel_cognitive + vel_social;
    }
}

void update_position(Particle *particle, double bounds[], int N)
{
    for (int i = 0; i < N; i++)
    {
        particle->position_i[i] = particle->position_i[i] + particle->velocity_i[i];

        if (particle->position_i[i] > bounds[i * N + 1])
        {
            particle->position_i[i] = bounds[i * N + 1];
        }

        if (particle->position_i[i] < bounds[i * N + 0])
        {
            particle->position_i[i] = bounds[i * N + 0];
        }
    }
}

void minimize(Function func, double x0[], double bounds[], int num_dimensions, int num_particle, int maxiter, int showResult)
{
    double *pos_best_g = (double *)malloc(sizeof(double) * num_dimensions);
    double err_best_g = -1;
    Particle *swarm = (Particle *)malloc(sizeof(Particle) * num_particle);

    for (int i = 0; i < num_particle; i++)
    {
        Particle particle;
        alloc_memory(&particle, num_dimensions);
        init_pso(&particle, num_dimensions, x0);
        swarm[i] = particle;
    }

    int i = 0;
    while (i < maxiter)
    {

        for (int j = 0; j < num_particle; j++)
        {
            evaluate(&swarm[j], func, num_dimensions);
            if (swarm[j].err_i < err_best_g || err_best_g == -1)
            {
                pos_best_g = swarm[j].position_i;
                err_best_g = swarm[j].err_i;
            }
        }

        for (int j = 0; j < num_particle; j++)
        {
            update_velocity(&swarm[j], pos_best_g, num_dimensions);
            update_position(&swarm[j], bounds, num_dimensions);
        }

        i++;

        if (showResult)
        {
            printf("Number Iteration: %d [%.20f , %.20f] error: %.20f\n", i, pos_best_g[0], pos_best_g[1], err_best_g);
        }
    }

    if (showResult)
    {
        printf("Final Solution: [%.20f , %.20f] error: %.20f\n", pos_best_g[0], pos_best_g[1], err_best_g);
    }
}

void pso_execute()
{
    double initial[] = {5, 5};
    int num_dimension = 2;
    int num_particle = 4096;
    int maxiter = 30;
    clock_t tempo_total_inicio = clock();
    minimize(sphere, initial, BOUNDS_SPHERE, num_dimension, num_particle, maxiter, false);
    clock_t tempo_total_fim = clock();
    double tempo_total = (double)(tempo_total_fim - tempo_total_inicio) / CLOCKS_PER_SEC / 1000;
    printf("\nTempo de execucao Tempo Total: %.6f ms\n", tempo_total);
}