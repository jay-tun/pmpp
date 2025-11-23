#include "filtering.h"

unsigned int compute_dim(std::uint64_t global_size, int block_size)
{
	return static_cast<unsigned int>((global_size / block_size) + (global_size % block_size > 0 ? 1 : 0));
}


__global__ void gray_scale_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h
)
{
	//1.2) Implement conversion
	std::uint64_t col = blockIdx.x*blockDim.x + threadIdx.x;
	std::uint64_t row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < w && row < h) {
		std::uint32_t pixel = src_data[row * w + col];

		unsigned char r = pixel & 0xff;
		unsigned char g = (pixel >> 8)  & 0xff;
		unsigned char b = (pixel >> 16) & 0xff;

		unsigned char gray = static_cast<unsigned char>(
			0.2126f*r + 0.7152f*g + 0.0722*b
		);
		dst_data[row * w + col] = (gray) | (gray << 8) | (gray << 16);
	}
}
void to_grayscale(gpu_image& dst, gpu_image const& src)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	gray_scale_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height);
}


__global__ void convolution_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h, 
	float* filter_data,
	std::uint64_t fw, std::uint64_t fh,
	bool use_abs_value
)
{
	//1.3) Implement convolution
	std::uint64_t col = blockIdx.x*blockDim.x + threadIdx.x;
	std::uint64_t row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= w || row >= h) return;

	int x_offset = fw / 2;
	int y_offset = fh / 2;

	float rf = 0.0f;
	float gf = 0.0f;
	float bf = 0.0f;

	for (int i = 0; i < (int) fw; ++i) {
		for (int j = 0; j < (int) fh; ++j) {
			int x = max(0, min(static_cast<int>(col - x_offset + i), static_cast<int>(w - 1)));
			int y = max(0, min(static_cast<int>(row - y_offset + j), static_cast<int>(h - 1)));

			uint32_t pixel = src_data[x + y * w];
			unsigned char r = pixel & 0xff;
			unsigned char g = (pixel >> 8) & 0xff;
			unsigned char b = (pixel >> 16) & 0xff;

			float weight = filter_data[i + j * fw];
			rf += r * weight;
			gf += g * weight;
			bf += b * weight;
		}
	}

	if(use_abs_value) {
		rf = abs(rf) / 2.f;
		gf = abs(gf) / 2.f;
		bf = abs(bf) / 2.f;
	}

	unsigned char rn = (unsigned char)min(max((int)(rf + 0.5f), 0), 255);
	unsigned char gn = (unsigned char)min(max((int)(gf + 0.5f), 0), 255);
	unsigned char bn = (unsigned char)min(max((int)(bf + 0.5f), 0), 255);

	dst_data[col + row * w] = (rn) | (gn << 8) | (bn << 16);
}
void apply_convolution(gpu_image& dst, gpu_image const& src, gpu_filter const& filter, bool use_abs_value)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	convolution_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height, filter.data.get(), filter.width, filter.height, use_abs_value);
}


constexpr int num_threads = 64;
__global__ void histogram_kernel(
	std::uint32_t* hist_data,
	std::uint32_t* img_data,
	std::uint64_t w, std::uint64_t h,
	std::uint8_t channel_flags = 1
)
{
	//1.4) Implement histogram computation
	__shared__ unsigned int local_hist[num_threads][num_bins];
	int t = threadIdx.x;
	int b = blockIdx.x;
	int idx = b * blockDim.x + t;

	//Memory initialization
	for (int i = 0; i < num_bins; ++i) {
		local_hist[t][i] = 0;
	}
	__syncthreads();

	for (uint64_t i = idx; i < w * h; i+= gridDim.x * blockDim.x) {
		uint32_t pixel = img_data[i];
		unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;
		//float value = 0.f;
		int bin_idx;
		
		if(channel_flags & 1) {
			bin_idx = r * num_bins / 256;
			atomicAdd(&local_hist[t][bin_idx], 1);
		}
		if(channel_flags & 2) {
			bin_idx = g * num_bins / 256;
			atomicAdd(&local_hist[t][bin_idx], 1);
		}
		if(channel_flags & 4) {
			bin_idx = b * num_bins / 256;
			atomicAdd(&local_hist[t][bin_idx], 1);
		}
	}
	__syncthreads();

	for (int i = 0; i < num_bins; i++)
	{
		atomicAdd(&hist_data[i], local_hist[t][i]);
	}
}
void compute_histogram(gpu_matrix<std::uint32_t>& hist, gpu_image const& img)
{
	std::uint8_t channel_flags = 1;
	cudaMemset(hist.data.get(), 0, hist.width * hist.height * sizeof(std::uint32_t));
	dim3 block_size = { num_threads };
	dim3 grid_size = { compute_dim(img.width, block_size.x) };
	histogram_kernel<<<grid_size, block_size>>>(hist.data.get(), img.data.get(), img.width, img.height, channel_flags);
}
