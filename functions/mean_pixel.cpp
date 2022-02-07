#include <stdint.h>
#include <vector>

std::vector<uint8_t> mean_pixel(uint8_t *img)
{

    uint32_t img_sum[3];
    uint32_t pixel_count = 0, color = 0;
    while(img)
    {
        img_sum[color] += *img;
        if(!color)
        {
            pixel_count++;
        }
        color = (color + 1) % 3;
        img++;
    }
    std::vector<uint8_t> img_mean(3);
    for (int8_t i = 0; i < 3; ++i)
    {
        img_mean[i] = static_cast<uint8_t>(img_sum[i] / pixel_count);
    }
    return img_mean;
}
