#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class Material;

struct HitRecord
{
    float t;
    Vec3 p;
    Vec3 normal;
    Material *mat_ptr;
};

class Hitable  {
    public:
        __host__ __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

#endif