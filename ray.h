#ifndef RAYH
#define RAYH
#include "vec3.h"

class Ray
{
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const Point3& a, const Vec3& b) { A = a; B = b; }
        __host__ __device__ Point3 origin() const       { return A; }
        __host__ __device__ Vec3 direction() const    { return B; }
        __host__ __device__ Point3 point_at_parameter(float t) const { return A + t*B; }

        Point3 A;
        Vec3 B;
};

#endif