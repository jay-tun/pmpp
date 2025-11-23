#pragma once

#include "image.h"

constexpr int num_bins = 32;

void to_grayscale(cpu_image& dst, cpu_image const& src);
void to_grayscale(gpu_image& dst, gpu_image const& src);

void apply_convolution(cpu_image& dst, cpu_image const& src, cpu_filter const& filter, bool use_abs_value = false);
void apply_convolution(gpu_image& dst, gpu_image const& src, gpu_filter const& filter, bool use_abs_value = false);

void compute_histogram(cpu_matrix<std::uint32_t>& hist, cpu_image const& img);
void compute_histogram(gpu_matrix<std::uint32_t>& hist, gpu_image const& img);

void draw_histogram(cpu_image const& img, cpu_matrix<std::uint32_t>& hist, std::uint32_t scale);