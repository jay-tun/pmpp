#pragma once

#include <cstdint>
#include <string>

#include "pointer.h"

template<typename T>
struct cpu_matrix
{
	std::uint64_t width;
	std::uint64_t height;

	host_ptr<T[]> data;
};

template<typename T>
struct gpu_matrix
{
	std::uint64_t width;
	std::uint64_t height;

	cuda_ptr<T[]> data;
};

template<typename T>
cpu_matrix<T> make_cpu_matrix(std::uint64_t width, std::uint64_t height);

template<typename T>
gpu_matrix<T> make_gpu_matrix(std::uint64_t width, std::uint64_t height);

template<typename T>
gpu_matrix<T> to_gpu(cpu_matrix<T> const& img);

template<typename T>
cpu_matrix<T> to_cpu(gpu_matrix<T> const& img);

using cpu_image = cpu_matrix<std::uint32_t>;
using gpu_image = gpu_matrix<std::uint32_t>;

cpu_image make_cpu_image(std::uint64_t width, std::uint64_t height);
gpu_image make_gpu_image(std::uint64_t width, std::uint64_t height);

cpu_image load(std::string const& path);
bool save(std::string const& path, cpu_image const& img);
bool save(std::string const& path, gpu_image const& img);

using cpu_filter = cpu_matrix<float>;
using gpu_filter = gpu_matrix<float>;

cpu_filter make_cpu_filter(std::uint64_t width, std::uint64_t height);
gpu_filter make_gpu_filter(std::uint64_t width, std::uint64_t height);