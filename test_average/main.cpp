#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

uint8_t* find_mean_pixel(uint8_t* img, int width, int height) {
    uint64_t r_sum = 0, g_sum = 0, b_sum = 0;
    uint64_t cnt = 0;
    for (int i = 0; i < width * height; i++) {
        r_sum += img[0];
        g_sum += img[1];
        b_sum += img[2];
        cnt++;
        img += 3;
    }
    uint8_t* result = new uint8_t[3];
    result[0] = r_sum / cnt;
    result[1] = g_sum / cnt;
    result[2] = b_sum / cnt;
    return result;
}

void find_average_jpg(const char* filename, const char* filename_out) {
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);

    uint8_t* mean_pix = find_mean_pixel(rgb_image, width, height);
    uint8_t* now = rgb_image;

    uint8_t* result = new uint8_t[width * height * 3];

    for (int i = 0; i < width * height; i++) {
        for (int j = 0; j < 3; j++) {
            if (now[j] > mean_pix[j])
                result[i * 3 + j] = now[j] - mean_pix[j];
            else
                result[i * 3 + j] = mean_pix[j] - now[j];
        }
        now += 3;
    }


	stbi_write_jpg(filename_out, width, height, 3, result, 100);
    stbi_image_free(rgb_image);
}

int main() {
    find_average_jpg("test.jpg", "out.jpg");
}
