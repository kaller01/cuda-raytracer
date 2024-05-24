#ifndef PIXELH
#define PIXELH
#include "vec3.h"

struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;

    __host__ __device__ Pixel() : r(0), g(0), b(0) {}
    __host__ __device__ Pixel(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}
    __host__ __device__ Pixel(Color3 v) : r(v.x()), g(v.y()), b(v.z()) {}
};

std::ostream &operator<<(std::ostream &os, const Pixel &pixel)
{
    os << static_cast<int>(pixel.r) << " " << static_cast<int>(pixel.g) << " " << static_cast<int>(pixel.b);
    return os;
}
#endif