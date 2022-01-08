#include <iostream>
#include <math.h>
using namespace std;

double BOUNDS_SPHERE[] = {-10, 10, -10, 10};
double BOUNDS_ROSENBROCK[] = {-2048, 2048, -2048, 2048};
double BOUNDS_RASTRINGIN[] = {-5.12, 5.12, -5.12, 5.12};
double BOUNDS_SCHWEFEL[] = {-500, 500, -500, 500};

typedef double (*Function)(double *x, int num_dimensions);

/**
 * @brief
 * https://www.cs.bham.ac.uk/research/projects/ecb/data/2/sphere.png
 * @param x
 * @return double
 */
double sphere(double *x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += pow(x[i], 2);
    }
    return total;
}

/**
 * @brief
 * https://upload.wikimedia.org/wikipedia/commons/3/32/Rosenbrock_function.svg
 * @param x
 * @return double
 */
double rosenbrock(double *x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += 100 * (pow(x[i], 2) - pow(x[i + 1], 2)) + pow(1 - x[i], 2);
    }
    return total;
}

/**
 * @brief
 * https://www.cs.bham.ac.uk/research/projects/ecb/data/1/rast.png
 * @param x
 * @return double
 */
double rastrigin(double *x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += (pow(x[i], 2) - (10 * cos(2 * 3.14159265359 * x[i])) + 10);
    }
    return total;
}

/**
 * @brief
 * https://www.sfu.ca/~ssurjano/schwef.png
 * @param x
 * @return double
 */
double schwefel(double *x, int N)
{
    double total = 0.0;
    for (int i = 0; i <= N; i++)
    {
        total += (x[i] * sin(sqrt(fabs(x[i]))));
    }
    return -total;
}

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

typedef struct Swarm
{
    Particle *swarm;
    double *swarm_best;
} Swarm;

void alloc_memory(Particle *particle, int num_dimensions)
{
    particle->position_i = (double *)malloc(sizeof(double) * num_dimensions);
    particle->velocity_i = (double *)malloc(sizeof(double) * num_dimensions);
    particle->pos_best_i = (double *)malloc(sizeof(double) * num_dimensions);
}

void init_pso(Particle *particle, int num_dimensions, double *x0, double *sbest)
{
    for (int i = 0; i < num_dimensions; i++)
    {
        particle->velocity_i[i] = RNG_UNIFORM(-1, 1);
        particle->position_i[i] = x0[i];
        particle->pos_best_i[i] = 0.0;
        sbest[i] = 0.0;
    }
}

Swarm *generate_swarms(int num_swarms, int num_particle, int num_dimensions, double *x0)
{

    Swarm *swarms = (Swarm *)malloc(sizeof(Swarm) * num_swarms);

    for (int k = 0; k < num_swarms; k++)
    {
        Particle *swarm = (Particle *)malloc(sizeof(Particle) * num_particle);
        swarms[k].swarm_best = (double *)malloc(sizeof(double) * num_dimensions);

        for (int i = 0; i < num_particle; i++)
        {
            Particle particle;
            alloc_memory(&particle, num_dimensions);
            init_pso(&particle, num_dimensions, x0, swarms[k].swarm_best);
            swarm[i] = particle;
        }
        swarms[k].swarm = swarm;
    }
    return swarms;
}

double *min_gbest(Swarm *swarms, int num_swarms, int num_dimensions)
{

    double *g_best = (double *)malloc(sizeof(double) * num_dimensions);
    g_best[0] = -1;
    g_best[1] = -1;

    for (int k = 0; k < num_swarms; k++)
    {
        if ((swarms[k].swarm_best[0] < g_best[0] && swarms[k].swarm_best[1] < g_best[1]) || g_best[0] == -1)
        {
            g_best = swarms[k].swarm_best;
        }
    }
    return g_best;
}

void print_single_swarm(Particle *swarm, int num_swarms, int num_particle)
{

    for (int k = 0; k < num_swarms; k++)
    {
        for (int i = k; i < num_particle; i++)
        {
            printf("I= %d Particle: (%f, %f) \n", (k + num_swarms * i), swarm[k + num_swarms * i].position_i[0], swarm[k + num_swarms * i].position_i[1]);
        }
    }
}

void prinf_mult_swarm(Swarm *swarms, int num_swarms, int num_particle, int num_dimension)
{
    for (int k = 0; k < num_swarms; k++)
    {
        printf("Swarm: %d \n", k);
        for (int i = 0; i < num_particle; i++)
        {
            for (int j = 0; j < num_dimension; j++)
            {
                printf("Particle: (%f)", swarms[k].swarm[i].position_i[j]);
            }
            printf("\n");
        }
    }
}

Particle *mergeSwarms(Swarm *swarms, int num_swarms, int num_particle)
{

    Particle *swarm = (Particle *)malloc(sizeof(Particle) * num_swarms * num_particle);

    for (int k = 0; k < num_swarms; k++)
    {
        for (int i = k; i < num_particle; i++)
        {
            swarm[k + num_swarms * i] = swarms[k].swarm[i];
        }
    }

    return swarm;
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
    double w = 0.9;
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

void minimize_mult(Function func, double x0[], double bounds[], Swarm *swarms, int num_swarms, int num_particle, int num_dimensions, int maxiter, int showResult)
{

    for (int k = 0; k < num_swarms; k++)
    {
        double *pos_best_s = (double *)malloc(sizeof(double) * num_dimensions);
        double err_best_s = -1;
        int i = 0;
        while (i < maxiter)
        {

            for (int j = 0; j < num_particle; j++)
            {
                evaluate(&swarms[k].swarm[j], func, num_dimensions);
                if (swarms[k].swarm[j].err_i < err_best_s || err_best_s == -1)
                {
                    pos_best_s = swarms[k].swarm[j].position_i;
                    err_best_s = swarms[k].swarm[j].err_i;
                }
            }

            for (int j = 0; j < num_particle; j++)
            {
                update_velocity(&swarms[k].swarm[j], pos_best_s, num_dimensions);
                update_position(&swarms[k].swarm[j], bounds, num_dimensions);
            }

            i++;
        }

        swarms[k].swarm_best[0] = pos_best_s[0];
        swarms[k].swarm_best[1] = pos_best_s[1];

        if (showResult)
        {
            printf("Final Solution swarm:%d: [%.20f , %.20f] \n", k, pos_best_s[0], pos_best_s[1], err_best_s);
        }
    }
}

void updateAllParticle(double bounds[], Swarm *swarms, int num_swarms, int num_particle, int num_dimensions)
{
    double *pos_best_g = min_gbest(swarms, num_swarms, num_dimensions);

    for (int k = 0; k < num_swarms; k++)
    {
        for (int j = 0; j < num_particle; j++)
        {
            double w = 0.4;
            int c1 = 1;
            int c2 = 2;
            int c3 = 1;
            for (int i = 0; i < num_dimensions; i++)
            {
                double r1 = RNG_RANDOM();
                double r2 = RNG_RANDOM();
                double r3 = RNG_RANDOM();

                double vel_cognitive = c1 * r1 * (swarms[k].swarm[j].pos_best_i[i] - swarms[k].swarm[j].position_i[i]);
                double vel_social = c2 * r2 * (pos_best_g[i] - swarms[k].swarm[j].position_i[i]);
                double vel_sbest = c3 * r3 * (swarms[k].swarm_best[i] - swarms[k].swarm[j].position_i[i]);
                swarms[k].swarm[j].velocity_i[i] = w * swarms[k].swarm[j].velocity_i[i] + vel_cognitive + vel_social + vel_sbest;

                swarms[k].swarm[j].position_i[i] = swarms[k].swarm[j].position_i[i] + swarms[k].swarm[j].velocity_i[i];

                if (swarms[k].swarm[j].position_i[i] > bounds[i * num_dimensions + 1])
                {
                    swarms[k].swarm[j].position_i[i] = bounds[i * num_dimensions + 1];
                }

                if (swarms[k].swarm[j].position_i[i] < bounds[i * num_dimensions + 0])
                {
                    swarms[k].swarm[j].position_i[i] = bounds[i * num_dimensions + 0];
                }
            }
        }
    }

    printf("Final Solution: [%.20f , %.20f] \n", pos_best_g[0], pos_best_g[1]);
}

int main()
{
    srand(time(0));
    double initial[] = {5, 5};
    int num_dimension = 2;
    int num_particle = 4096;
    int num_swarms = 4;
    int maxiter = 30;

    clock_t tempo_total_inicio = clock();
    Swarm *swarms = generate_swarms(num_swarms, num_particle, num_dimension, initial);
    minimize_mult(sphere, initial, BOUNDS_SPHERE, swarms, num_swarms, num_particle, num_dimension, maxiter, true);
    updateAllParticle(BOUNDS_SPHERE, swarms, num_swarms, num_particle, num_dimension);

    clock_t tempo_total_fim = clock();
    double tempo_total = (double)(tempo_total_fim - tempo_total_inicio);
    printf("Time elapsed on CPU: %3.2f ms.\n", tempo_total);

    return 0;
}
