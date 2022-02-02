#include <stdint.h>
#include <algorithm>

int8_t pixel_distance(int8_t* x, int8_t* y) {
    int32_t r = (int32_t)(x[0]) - y[0];
    int32_t g = (int32_t)(x[1]) - y[1];
    int32_t b = (int32_t)(x[2]) - y[2];

    int32_t sqr_distance = r * r + g * g + b * b;
    return (int8_t)std::min(255.0, floor(sqrt(sqr_distance)));
}
