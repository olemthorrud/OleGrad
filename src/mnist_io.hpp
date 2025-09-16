#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>


struct MnistImages {
  std::size_t count = 0;
  std::size_t rows  = 0;
  std::size_t cols  = 0;
  std::vector<double> data;
};

struct MnistLabels {
  std::vector<uint8_t> y;
};


inline uint32_t read_u32_be(std::ifstream& f) {
  uint8_t b[4];
  if (!f.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("Unexpected EOF");
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}


inline MnistImages load_mnist_images(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open: " + path);

  const uint32_t magic = read_u32_be(f);
  if (magic != 2051u) throw std::runtime_error("Bad magic (images): " + std::to_string(magic));

  const uint32_t num  = read_u32_be(f);
  const uint32_t rows = read_u32_be(f);
  const uint32_t cols = read_u32_be(f);

  MnistImages out;
  out.count = num;
  out.rows  = rows;
  out.cols  = cols;
  out.data.resize(std::size_t(num) * rows * cols);

  
  std::vector<uint8_t> buf(out.data.size());
  if (!f.read(reinterpret_cast<char*>(buf.data()), buf.size()))
    throw std::runtime_error("Unexpected EOF (image data)");

  for (std::size_t i = 0; i < buf.size(); ++i)
    out.data[i] = static_cast<double>(buf[i]) / 255.0;

  return out;
}

inline MnistLabels load_mnist_labels(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open: " + path);

  const uint32_t magic = read_u32_be(f);
  if (magic != 2049u) throw std::runtime_error("Bad magic (labels): " + std::to_string(magic));

  const uint32_t num = read_u32_be(f);
  MnistLabels out;
  out.y.resize(num);

  if (!f.read(reinterpret_cast<char*>(out.y.data()), out.y.size()))
    throw std::runtime_error("Unexpected EOF (labels)");
  return out;
}

inline const double* image_ptr(const MnistImages& X, std::size_t i) {
  return &X.data[i * X.rows * X.cols];
}

void show_image(const MnistImages& X, const MnistLabels& Y, std::size_t i) {
    const double* img = image_ptr(X, i);
    std::cout << "Label: " << int(Y.y[i]) << "\n";
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) {
            double v = img[r * X.cols + c];
            char ch = v > 0.5 ? '#' : (v > 0.2 ? '+' : '.');
            std::cout << ch;
        }
        std::cout << "\n";
    }
}