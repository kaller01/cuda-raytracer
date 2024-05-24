#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class Vec3  {
public:
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float e0, float e1, float e2) : e0{e0}, e1{e1}, e2{e2} {}
    __host__ __device__ Vec3(float e) : e0{e}, e1{e}, e2{e} {}
    __host__ __device__ inline float x() const { return e0; }
    __host__ __device__ inline float y() const { return e1; }
    __host__ __device__ inline float z() const { return e2; }
    __host__ __device__ inline float r() const { return e0; }
    __host__ __device__ inline float g() const { return e1; }
    __host__ __device__ inline float b() const { return e2; }
    __host__ __device__ inline float& x() { return e0; }
    __host__ __device__ inline float& y() { return e1; }
    __host__ __device__ inline float& z() { return e2; }
    __host__ __device__ inline float& r() { return e0; }
    __host__ __device__ inline float& g() { return e1; }
    __host__ __device__ inline float& b() { return e2; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e0, -e1, -e2); }

    __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e0*e0 + e1*e1 + e2*e2); }
    __host__ __device__ inline float squared_length() const { return e0*e0 + e1*e1 + e2*e2; }
    __host__ __device__ inline void make_unit_vector();

    float e0;
    float e1;
    float e2;
};


using Point3 = Vec3;
using Color3 = Vec3;

inline std::istream& operator>>(std::istream &is, Vec3 &t) {
    is >> t.e0 >> t.e1 >> t.e2;
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const Vec3 &t) {
    os << t.e0 << " " << t.e1 << " " << t.e2;
    return os;
}

__host__ __device__ inline void Vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e0*e0 + e1*e1 + e2*e2);
    e0 *= k; e1 *= k; e2 *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e0 + v2.e0, v1.e1 + v2.e1, v1.e2 + v2.e2);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e0 - v2.e0, v1.e1 - v2.e1, v1.e2 - v2.e2);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e0 * v2.e0, v1.e1 * v2.e1, v1.e2 * v2.e2);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e0 / v2.e0, v1.e1 / v2.e1, v1.e2 / v2.e2);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t*v.e0, t*v.e1, t*v.e2);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return Vec3(v.e0/t, v.e1/t, v.e2/t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(t*v.e0, t*v.e1, t*v.e2);
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2) {
    return v1.e0 *v2.e0 + v1.e1 *v2.e1  + v1.e2 *v2.e2;
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3( (v1.e1*v2.e2 - v1.e2*v2.e1),
                (-(v1.e0*v2.e2 - v1.e2*v2.e0)),
                (v1.e0*v2.e1 - v1.e1*v2.e0));
}


__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3 &v){
    e0  += v.e0;
    e1  += v.e1;
    e2  += v.e2;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3 &v){
    e0  *= v.e0;
    e1  *= v.e1;
    e2  *= v.e2;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3 &v){
    e0  /= v.e0;
    e1  /= v.e1;
    e2  /= v.e2;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
    e0  -= v.e0;
    e1  -= v.e1;
    e2  -= v.e2;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
    e0  *= t;
    e1  *= t;
    e2  *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    float k = 1.0/t;

    e0  *= k;
    e1  *= k;
    e2  *= k;
    return *this;
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

#endif