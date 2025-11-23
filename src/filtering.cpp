#include "filtering.h"

void to_grayscale(cpu_image& dst, cpu_image const& src)
{
	auto w = std::min(src.width, dst.width);
	auto h = std::min(src.height, dst.height);

	for(std::uint64_t x_index = 0; x_index < w; ++x_index)
	{
		for(std::uint64_t y_index = 0; y_index < h; ++y_index)
		{
			auto pixel = src.data[x_index + y_index * w];
			unsigned char r = pixel & 0xff;
			unsigned char g = (pixel >> 8) & 0xff;
			unsigned char b = (pixel >> 16) & 0xff;

			float f = 0.2126f * static_cast<float>(r) + 0.7152f * static_cast<float>(g) + 0.0722f * static_cast<float>(b);
			unsigned char l = static_cast<unsigned char>(f + 0.5f);

			dst.data[x_index + y_index * w] = (l) | (l << 8) | (l << 16);
		}
	}
}


void apply_convolution(cpu_image& dst, cpu_image const& src, cpu_filter const& filter, bool use_abs_value)
{
	auto w = std::min(src.width, dst.width);
	auto h = std::min(src.height, dst.height);

	for(int x_index = 0; x_index < static_cast<int>(w); ++x_index)
	{
		for(int y_index = 0; y_index < static_cast<int>(h); ++y_index)
		{
			int x_offset = static_cast<int>(filter.width) / 2;
			int y_offset = static_cast<int>(filter.height) / 2;

			float rf = 0.f;
			float gf = 0.f;
			float bf = 0.f;

			for(int i = 0; i < static_cast<int>(filter.width); ++i)
			{
				for(int j = 0; j < static_cast<int>(filter.height); ++j)
				{
					auto x = std::max(0, std::min((x_index - x_offset + i), static_cast<int>(w - 1)));
					auto y = std::max(0, std::min((y_index - y_offset + j), static_cast<int>(h - 1)));
					auto pixel = src.data[x + y * w];
					unsigned char r = pixel & 0xff;
					unsigned char g = (pixel >> 8) & 0xff;
					unsigned char b = (pixel >> 16) & 0xff;

					float weight = filter.data[i + j * filter.width];
					rf += static_cast<float>(r) * weight;
					gf += static_cast<float>(g) * weight;
					bf += static_cast<float>(b) * weight;
				}
			}

			if(use_abs_value)
			{
				rf = abs(rf) / 2.f;
				gf = abs(gf) / 2.f;
				bf = abs(bf) / 2.f;
			}

			unsigned char rn = static_cast<unsigned char>(rf + 0.5f);
			unsigned char gn = static_cast<unsigned char>(gf + 0.5f);
			unsigned char bn = static_cast<unsigned char>(bf + 0.5f);

			dst.data[x_index + y_index * w] = (rn) | (gn << 8) | (bn << 16);
		}
	}

}


void compute_histogram(cpu_matrix<std::uint32_t>& hist, cpu_image const& img)
{
	std::uint8_t channel_flags = 1;

	auto w = img.width;
	auto h = img.height;

	for(std::uint64_t x_index = 0; x_index < w; ++x_index)
	{
		for(std::uint64_t y_index = 0; y_index < h; ++y_index)
		{
			auto pixel = img.data[x_index + y_index * w];
			unsigned char r = pixel & 0xff;
			unsigned char g = (pixel >> 8) & 0xff;
			unsigned char b = (pixel >> 16) & 0xff;

			if(channel_flags & 1)
				hist.data[r / (256 / num_bins)] += 1;
			if(channel_flags & 2)
				hist.data[g / (256 / num_bins)] += 1;
			if(channel_flags & 4)
				hist.data[b / (256 / num_bins)] += 1;
		}
	}
}

void draw_histogram(cpu_image const& img, cpu_matrix<std::uint32_t>& hist, std::uint32_t scale)
{
	auto col = img.width / num_bins;
	
	for(std::uint64_t i = 0; i < num_bins; ++i)
	{
		for(std::uint64_t x_index = i * col; x_index < std::min(img.width, (i+1)*col); ++x_index)
		{
			auto h = std::min(static_cast<std::uint32_t>(img.height), hist.data[i] / scale);
			for(std::uint64_t y_index = 0; y_index < h; ++y_index)
				img.data[x_index + (img.height - y_index - 1) * img.width] = (250) | (180 << 8) | (33 << 16);
			for(std::uint64_t y_index = h; y_index < img.height; ++y_index)
				img.data[x_index + (img.height - y_index - 1) * img.width] = (50) | (50 << 8) | (50 << 16);
		}
	}
}