#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list: public Hitable  {
    public:
        __host__ __device__ hitable_list() {}
        __host__ __device__ hitable_list(Hitable **l, int n) {list = l; list_size = n; }
        __host__ __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
        Hitable **list;
        int list_size;
};

__host__ __device__ bool hitable_list::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif