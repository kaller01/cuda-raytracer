// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Ray tracer utils
#include "pixel.h"
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "float.h"
#include "camera.h"
#include "material.h"
#include "world.h"

#define RAYS_PER_PIXEL 64
#define STEPS_PER_RAY 64
#define HITABLES (22*22 + 1 + 4)
#define MAX_CONST_MEMORY 8000
#define CONST_MEMORY_POOL_SIZE (MAX_CONST_MEMORY - HITABLES)/2

__constant__ Hitable *d_list_constant[HITABLES];

__constant__ void *const_hitables[CONST_MEMORY_POOL_SIZE];

__constant__ void *const_materials[CONST_MEMORY_POOL_SIZE];

__device__ Color3 trace_color(const Ray &r, Hitable **world, curandState *local_rand_state)
{
    Ray cur_ray = r;
    Color3 attenuation_total = Color3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < STEPS_PER_RAY; i++)
    {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            Color3 attenuation_hit_material;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation_hit_material, scattered, local_rand_state))
            {
                attenuation_total *= attenuation_hit_material;
                cur_ray = scattered;
            }
            else
            {
                // No scatter, all light absorbed
                return rec.mat_ptr->emitted();
            }
        }
        else
        {
            // No hit, means we hit the "sky".
            Vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            // Sky is a faded color
            Color3 sky = (1.0f - t) * Color3(1.0) + t * Color3(0.5, 0.7, 1.0);
            // sky *= 0.1;
            // Return sky with total attentuation.
            return attenuation_total * sky;
        }
    }
    // Exceeded recursion, no light found, returning black.
    return Color3(0.0, 0.0, 0.0);
}

__global__ void render(Pixel *frame_buffer, int max_x, int max_y, Camera **cam, Hitable **world, curandState *rand_state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y))
        return;
    int pixel_index = y * max_x + x;

    curandState local_rand_state = rand_state[pixel_index]; // Get random state for this pixel
    Color3 attenuation_sum{0, 0, 0};
    for (int s = 0; s < RAYS_PER_PIXEL; s++)
    {
        float u = float(x + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(y + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        attenuation_sum += trace_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state; // we have used the state, save it

    Color3 color = attenuation_sum / float(RAYS_PER_PIXEL);
    color.r() = sqrt(max(color.r(), 0.0f));
    color.g() = sqrt(max(color.g(), 0.0f));
    color.b() = sqrt(max(color.b(), 0.0f));

    frame_buffer[pixel_index] = Pixel(color * 255);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state, Hitable **d_world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_world = new hitable_list(d_list_constant, HITABLES);
    }
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void move_material_pointer_to_const(Hitable **d_list, void** tmp_materials)
{
    Material *base_tmp = (Material *)tmp_materials;
    Material *base_const = (Material *)const_materials;
    for (size_t i = 0; i < HITABLES; i++)
    {
        ((Sphere*) d_list[i])->mat_ptr = ((Sphere*) d_list[i])->mat_ptr - base_tmp + base_const;
    }
}

__global__ void move_sphere_pointer_to_const(void **tmp_hitables, Hitable **d_list)
{
    Hitable *base_tmp = (Hitable *)tmp_hitables;
    Hitable *base_const = (Hitable *)const_hitables;
    for (size_t i = 0; i < HITABLES; i++)
    {
        d_list[i] = d_list[i] - base_tmp + base_const;        
    }
}

int main()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cerr << "\n=== CONST MEMORY STATS ===\n";
    std::cerr << "GPU const memory : " << deviceProp.totalConstMem << " bytes\n";
    std::cerr << "Sphere size: " << sizeof(Sphere) << " bytes \n";
    std::cerr << "Largest material size: " << sizeof(Metal) << " bytes\n";
    std::cerr << "Total const memory required for "<< HITABLES << " hitables: " << HITABLES * sizeof(Sphere) + HITABLES * sizeof(Metal) + HITABLES * sizeof(Hitable*) << "\n";

    int tx = 16;
    int ty = 16;    
    int nx = 800;
    int ny = 400;
    int num_pixels = nx * ny;
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    std::cerr << "\n=== FRAME STATS ===\n";
    std::cerr << "Rendering a " << nx << "x" << ny << " image "<< "in " << tx << "x" << ty << " blocks.\n";
    std::cerr << "Memory usage per pixel: " << sizeof(Pixel) << " bytes \n";
    std::cerr << "Frame buffer size: " << num_pixels * sizeof(Pixel) << " bytes \n";

    Camera **d_camera;
    checkCudaErrors(cudaMallocManaged((void **)&d_camera, sizeof(Camera *)));
    Hitable **d_list;
    checkCudaErrors(cudaMallocManaged((void **)&d_list, HITABLES * sizeof(Hitable *)));
    Hitable **d_world;
    checkCudaErrors(cudaMallocManaged((void **)&d_world, sizeof(Hitable *)));

    // init world, with some randomness
    {   
        void **tmp_hitables;
        checkCudaErrors(cudaMallocManaged((void **)&tmp_hitables, CONST_MEMORY_POOL_SIZE * sizeof(void *)));
        void **tmp_materials;
        checkCudaErrors(cudaMallocManaged((void **)&tmp_materials, CONST_MEMORY_POOL_SIZE * sizeof(void *)));

        {
            curandState *curand_state_world;
            {   // init rand state
                checkCudaErrors(cudaMalloc((void **)&curand_state_world, 1 * sizeof(curandState)));
                rand_init<<<1, 1>>>(curand_state_world);
                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }
            // create world
            create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, curand_state_world, tmp_hitables, tmp_materials);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            // free
            checkCudaErrors(cudaFree(curand_state_world));
        }

        {   // move materials to const memory
            cudaMemcpyToSymbol(const_materials, tmp_materials, CONST_MEMORY_POOL_SIZE * sizeof(void *));
            move_material_pointer_to_const<<<1,1>>>(d_list, tmp_materials);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
        checkCudaErrors(cudaFree(tmp_materials));
        
        {   // move hitables to const memory
            cudaMemcpyToSymbol(const_hitables, tmp_hitables, CONST_MEMORY_POOL_SIZE * sizeof(void *));
            move_sphere_pointer_to_const<<<1, 1>>>(tmp_hitables, d_list);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
        checkCudaErrors(cudaFree(tmp_hitables));

        {   // move d_list to const
            cudaMemcpyToSymbol(d_list_constant, d_list, HITABLES * sizeof(Hitable *));
        }
        checkCudaErrors(cudaFree(d_list));
    }

    curandState *curand_state;
    {
        checkCudaErrors(cudaMalloc((void **)&curand_state, num_pixels * sizeof(curandState)));
        render_init<<<blocks, threads>>>(nx, ny, curand_state, d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // allocate frame_buffer
    Pixel *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer,  num_pixels * sizeof(Pixel)));

    std::cerr << "\n=== RENDER ===\n";
    clock_t start, stop;
    start = clock();
    render<<<blocks, threads>>>(frame_buffer, nx, ny, d_camera, d_world, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(curand_state));
    
    // Output frame_buffer as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            Pixel pixel = frame_buffer[j * nx + i];
            std::cout << pixel << "\n";
        }
    }
    checkCudaErrors(cudaFree(frame_buffer));
}