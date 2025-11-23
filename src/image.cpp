#include "image.h"

#include <fstream>
#include <limits>
#include <cuda_runtime.h>

template<typename T>
cpu_matrix<T> make_cpu_matrix(std::uint64_t width, std::uint64_t height)
{
	cpu_matrix<T> img;
	img.width = width;
	img.height = height;
	//1.1a) Allocate host memory
	img.data = make_host_array<T>(width * height);
	return img;
}

template cpu_matrix<std::uint32_t> make_cpu_matrix(std::uint64_t width, std::uint64_t height);
template cpu_matrix<float> make_cpu_matrix(std::uint64_t width, std::uint64_t height);


template<typename T>
gpu_matrix<T> make_gpu_matrix(std::uint64_t width, std::uint64_t height)
{
	gpu_matrix<T> img;
	img.width = width;
	img.height = height;
	//1.1a) Allocate device memory
	img.data = make_managed_cuda_array<T>(width * height);
	return img;
}

template gpu_matrix<std::uint32_t> make_gpu_matrix(std::uint64_t width, std::uint64_t height);
template gpu_matrix<float> make_gpu_matrix(std::uint64_t width, std::uint64_t height);


template<typename T>
gpu_matrix<T> to_gpu(cpu_matrix<T> const& img)
{
	gpu_matrix<T> cpy;
	cpy.width = img.width;
	cpy.height = img.height;
	//1.1b) Allocate and copy to device memory
	//Allocate
	cpy.data = make_managed_cuda_array<T>(img.width * img.height);
	// if (!cpy.data) throw std::bad_alloc();

	//Copy to device
	auto err = cudaMemcpy(cpy.data.get(),
						 img.data.get(), 
						 sizeof(T) * img.width * img.height,
						 cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(err));
	}
	
	return cpy;
}

template gpu_matrix<std::uint32_t> to_gpu(cpu_matrix<std::uint32_t> const& img);
template gpu_matrix<float> to_gpu(cpu_matrix<float> const& img);


template<typename T>
cpu_matrix<T> to_cpu(gpu_matrix<T> const& img)
{
	cpu_matrix<T> cpy;
	cpy.width = img.width;
	cpy.height = img.height;
	//1.1b) Allocate and copy to host memory
	//Allocate
	cpy.data = make_host_array<T>(img.width * img.height);
	
	//Copy to host
	auto err = cudaMemcpy(cpy.data.get(),
						  img.data.get(),
						  sizeof(T) * img.width * img.height,
						  cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(err));
	}
	return cpy;
}

template cpu_matrix<std::uint32_t> to_cpu(gpu_matrix<std::uint32_t> const& img);
template cpu_matrix<float> to_cpu(gpu_matrix<float> const& img);


cpu_image make_cpu_image(std::uint64_t width, std::uint64_t height)
{
	return make_cpu_matrix<std::uint32_t>(width, height);
}

gpu_image make_gpu_image(std::uint64_t width, std::uint64_t height)
{
	return make_gpu_matrix<std::uint32_t>(width, height);
}

cpu_image load(std::string const& path)
{
	cpu_image img;

	if(path.empty())
		throw std::runtime_error("Invalid filename");

	std::ifstream is(path, std::ifstream::binary);
	if(is.fail())
		throw std::runtime_error("Could not open file");

	char line[256];
	is.getline(line, sizeof(line));

	if(line[0] != 'P' || line[1] != '6')
		throw std::runtime_error("Invalid identification string");

	// Skip comment
	while(is.peek() == '#')
		is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	is >> img.width; is >> img.height;
	img.data = make_host_array<std::uint32_t>(img.width * img.height);

	unsigned short max;
	is >> max;
	if(max > 255)
		throw std::runtime_error("max > 255 is unsupported");

	is.ignore(1, '\n');

	// Read pixels into temporary buffer
	auto tmp = make_host_array<unsigned char>(img.width * img.height * 3);
	is.read(reinterpret_cast<char*>(tmp.get()), img.width * img.height * 3);

	// Convert RGB to RGBA
	for(std::uint64_t y = 0; y < img.height; y++) {
		for(std::uint64_t x = 0; x < img.width; x++) {
			std::uint64_t idx = y * img.width * 3 + x * 3;
			unsigned char r = tmp[idx], g = tmp[idx + 1], b = tmp[idx + 2];
			img.data[y * img.width + x] = r | (g << 8) | (b << 16);
		}
	}

	return img;
}

bool save(std::string const& path, cpu_image const& img)
{
	if(path.empty())
		throw std::runtime_error("Invalid filename");

	std::ofstream os(path, std::ofstream::binary);
	if(os.fail())
		throw std::runtime_error("Could not open file");

	os << "P6" << std::endl;
	os << img.width << std::endl;
	os << img.height << std::endl;
	os << "255" << std::endl;

	// Convert RGBA to RGB
	auto tmp = make_host_array<unsigned char>(img.width * img.height * 3);
	for(std::uint64_t y = 0; y < img.height; y++) {
		for(std::uint64_t x = 0; x < img.width; x++) {
			std::uint64_t idx = y * img.width * 3 + x * 3;
			std::uint32_t rgba = img.data[y * img.width + x];
			unsigned char r = rgba & 0xff, g = (rgba >> 8) & 0xff, b = (rgba >> 16) & 0xff;
			tmp[idx] = r; tmp[idx + 1] = g; tmp[idx + 2] = b;
		}
	}
	os.write(reinterpret_cast<const char*>(tmp.get()), img.width * img.height * 3);
	return true;
}

bool save(std::string const& path, gpu_image const& img)
{
	if(path.empty())
		throw std::runtime_error("Invalid filename");

	return save(path, to_cpu(img));
}

cpu_filter make_cpu_filter(std::uint64_t width, std::uint64_t height)
{
	return make_cpu_matrix<float>(width, height);
}

gpu_filter make_gpu_filter(std::uint64_t width, std::uint64_t height)
{
	return make_gpu_matrix<float>(width, height);
}
