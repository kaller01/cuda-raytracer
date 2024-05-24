#ifndef WORLDH
#define WORLDH

#define RND (curand_uniform(&local_rand_state))

__device__ void **addr(void **&tmp_const_memory_pool, size_t size)
{
    void **addr = tmp_const_memory_pool;
    //size is in bytes, += 1 jumps one void** (8 bytes)
    tmp_const_memory_pool += size/8;
    return addr;
}

#define NEW(a) new (addr(tmp_hitables, sizeof(a))) a
#define NEWM(a) new (addr(tmp_materials, sizeof(a))) a
#define X 11

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, void **tmp_hitables, void **tmp_materials)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = NEW(Sphere)(Point3(0, -1000.0, -1), 1000, NEWM(Lambertian)(Color3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -X; a < X; a++)
        {
            for (int b = -X; b < X; b++)
            {
                float choose_mat = RND;
                Vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.3f)
                {
                    d_list[i++] = NEW(Sphere)(center, 0.2, NEWM(Lambertian)(Color3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.7f)
                {
                    d_list[i++] = NEW(Sphere)(center, 0.2, NEWM(Metal)(Color3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else
                {
                    d_list[i++] = NEW(Sphere)(center, 0.2, NEWM(DiffuseLight)(Color3(RND * RND, RND * RND, RND * RND)));
                }
            }
        }
        d_list[i++] = NEW(Sphere)(Point3(0, 1, 0), 1.0, NEWM(Dielectric)(1.5));
        d_list[i++] = NEW(Sphere)(Point3(-4, 1, 0), 1.0, NEWM(DiffuseLight)(Color3(1, 0.9, 0.9)));
        d_list[i++] = NEW(Sphere)(Point3(4, 1, 0), 1.0, NEWM(Metal)(Color3(0.7, 0.6, 0.5), 0.0));
        d_list[i++] = NEW(Sphere)(Point3(50, 20, 0), 5.0, NEWM(DiffuseLight)(Color3(1, 1, 0)));

        *rand_state = local_rand_state;

        Vec3 lookfrom(13, 2, 3);
        Vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new Camera(lookfrom,
                               lookat,
                               Vec3(0, 1, 0),
                               30.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

#endif