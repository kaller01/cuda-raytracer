#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class Sphere: public Hitable  {
    public:
        __device__ __host__ Sphere() {}
        __device__ __host__ Sphere(Vec3 cen, float r, Material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ __host__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
        Vec3 center;
        float radius;
        Material *mat_ptr;
};

__device__ __host__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif