#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <libheif/heif.h>

using namespace std;

struct File_info {
	const uint8_t* pixel_data;
	int height;
	int width;
};

File_info read_heic(const char* filename) {
	heif_context* ctx = heif_context_alloc();
	heif_context_read_from_file(ctx, filename, nullptr);

	// get a handle to the primary image
	heif_image_handle* handle;
	heif_context_get_primary_image_handle(ctx, &handle);

	// decode the image and convert colorspace to RGB, saved as 24bit interleaved
	heif_image* img;
	heif_decode_image(handle, &img, heif_colorspace_RGB, heif_chroma_interleaved_RGB, nullptr);

	int stride;
	const uint8_t* data = heif_image_get_plane_readonly(img, heif_channel_interleaved, &stride);

	int height = heif_image_handle_get_height(handle),
		width = heif_image_handle_get_width(handle);

	File_info f({ data, height, width });

	return f;
}

int main(int argc, char** argv) {

	const char* input_filename = "C:\\Users\\Nexei\\Downloads\\IMG_2056.HEIC";

	const char* output_filename = "C:\\Users\\Nexei\\Downloads\\out_test.jpg";

	File_info info = read_heic(input_filename);

	stbi_write_jpg(output_filename, info.width, info.height, 3, info.pixel_data, 100);

	return 0;
}