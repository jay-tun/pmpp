
#include "image.h"
#include "pointer.h"
#include "filtering.h"

#include <format>

enum class log_level {INFO, WARNING, ERROR};
void log(log_level lvl, std::string const & s)
{
	std::fputs(s.c_str(), lvl == log_level::ERROR ? stderr : stdout);
	std::fflush(lvl == log_level::ERROR ? stderr : stdout);
}

cpu_image create_grayscale(cpu_image const& src)
{
	auto gray_cpu = make_cpu_image(src.width, src.height);
	to_grayscale(gray_cpu, src);
	return gray_cpu;
}
gpu_image create_grayscale(gpu_image const& src)
{
	auto gray_gpu = make_gpu_image(src.width, src.height);
	auto err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error allocating image: '{}'\n", cudaGetErrorString(err)));

	to_grayscale(gray_gpu, src);
	err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error executing grayscale kernel: '{}'\n", cudaGetErrorString(err)));

	return gray_gpu;
}

cpu_filter create_filter(bool horizontal)
{
	auto filter_cpu = make_cpu_filter(3, 3);
	if(horizontal)
	{
		filter_cpu.data[0] =  1.f;
		filter_cpu.data[1] =  0.f;
		filter_cpu.data[2] = -1.f;
		filter_cpu.data[3] =  2.f;
		filter_cpu.data[4] =  0.f;
		filter_cpu.data[5] = -2.f;
		filter_cpu.data[6] =  1.f;
		filter_cpu.data[7] =  0.f;
		filter_cpu.data[8] = -1.f;
	}
	else
	{
		filter_cpu.data[0] =  1.f;
		filter_cpu.data[1] =  2.f;
		filter_cpu.data[2] =  1.f;
		filter_cpu.data[3] =  0.f;
		filter_cpu.data[4] =  0.f;
		filter_cpu.data[5] =  0.f;
		filter_cpu.data[6] = -1.f;
		filter_cpu.data[7] = -2.f;
		filter_cpu.data[8] = -1.f;
	}
	return filter_cpu;
}
cpu_image create_edgedetect(cpu_image const& src, bool horizontal)
{
	auto filter_cpu = create_filter(horizontal);
	auto filtered_cpu = make_cpu_image(src.width, src.height);
	apply_convolution(filtered_cpu, src, filter_cpu, true);
	return filtered_cpu;
}
gpu_image create_edgedetect(gpu_image const& src, bool horizontal)
{
	auto filter_cpu = create_filter(horizontal);
	auto filter_gpu = to_gpu(filter_cpu);
	auto err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error copying image: '{}'\n", cudaGetErrorString(err)));

	auto filtered_gpu = make_gpu_image(src.width, src.height);
	err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error allocating image: '{}'\n", cudaGetErrorString(err)));

	apply_convolution(filtered_gpu, src, filter_gpu, true);
	err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error executing convolution kernel: '{}'\n", cudaGetErrorString(err)));

	return filtered_gpu;
}

cpu_matrix<std::uint32_t> create_histogram(cpu_image const& img)
{
	auto hist_cpu = make_cpu_matrix<std::uint32_t>(num_bins, 1);
	compute_histogram(hist_cpu, img);
	
	return hist_cpu;
}
gpu_matrix<std::uint32_t> create_histogram(gpu_image const& img)
{
	auto hist_gpu = make_gpu_matrix<std::uint32_t>(num_bins, 1);
	auto err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error allocating histogram: '{}'\n", cudaGetErrorString(err)));

	compute_histogram(hist_gpu, img);
	err = cudaGetLastError();
	if(err)
		log(log_level::ERROR, std::format("Error computing histogram: '{}'\n", cudaGetErrorString(err)));

#if 0
	auto hist_cpu = to_cpu(hist_gpu);
	std::uint64_t sum = 0;
	for(std::uint64_t b = 0; b < hist_cpu.width; ++b)
	{
		log(log_level::INFO, std::format("{} {}\n", b, hist_cpu.data[b]));
		sum += hist_cpu.data[b];
	}
	log(log_level::INFO, std::format("Total histogram pixels: {}\n", sum));
#endif
	return hist_gpu;
}

int main()
{
	log(log_level::INFO, "Loading image...");
	cpu_image base_cpu = load("img/Cornell_Box_with_3_balls_of_different_materials.ppm");
	save("out/cornell_cpu_unchanged.ppm", base_cpu);
	log(log_level::INFO, "CPU...");
	auto base_gpu = to_gpu(base_cpu);
	save("out/cornell_gpu_unchanged.ppm", base_gpu);
	log(log_level::INFO, "GPU\n");
	
	log(log_level::INFO, "Converting image to grayscale...");
	auto gray_cpu = create_grayscale(base_cpu);
	save("out/cornell_cpu_grayscale.ppm", gray_cpu);
	log(log_level::INFO, "CPU...");
	auto gray_gpu = create_grayscale(base_gpu);
	save("out/cornell_gpu_grayscale.ppm", gray_gpu);
	log(log_level::INFO, "GPU\n");
	
	
	log(log_level::INFO, "Filtering image (h)...");
	auto filtered_h_cpu = create_edgedetect(gray_cpu, true);
	save("out/cornell_cpu_filtered_h.ppm", filtered_h_cpu);
	log(log_level::INFO, "CPU...");
	auto filtered_h_gpu = create_edgedetect(gray_gpu, true);
	save("out/cornell_gpu_filtered_h.ppm", filtered_h_gpu);
	log(log_level::INFO, "GPU\n");

	log(log_level::INFO, "Filtering image (v)...");
	auto filtered_v_cpu = create_edgedetect(gray_cpu, false);
	save("out/cornell_cpu_filtered_v.ppm", filtered_v_cpu);
	log(log_level::INFO, "CPU...");
	auto filtered_v_gpu = create_edgedetect(gray_gpu, false);
	save("out/cornell_gpu_filtered_v.ppm", filtered_v_gpu);
	log(log_level::INFO, "GPU\n");
	
	log(log_level::INFO, "Computing histogram...");
	auto hist_cpu = create_histogram(base_cpu);
	auto plot_cpu = make_cpu_image(1024, 1024);
	draw_histogram(plot_cpu, hist_cpu, 1200);
	save("out/cornell_cpu_histogram.ppm", plot_cpu);
	log(log_level::INFO, "CPU...");
	auto hist_gpu = create_histogram(base_gpu);
	auto hist_gpu_transfered = to_cpu(hist_gpu);
	auto plot_gpu = make_cpu_image(1024, 1024);
	draw_histogram(plot_gpu, hist_gpu_transfered, 1200);
	save("out/cornell_gpu_histogram.ppm", plot_gpu);
	log(log_level::INFO, "GPU\n");

	return 0;
}