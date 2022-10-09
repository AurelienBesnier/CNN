#ifndef __LAYERCONVPOOL_HPP__
#define __LAYERCONVPOOL_HPP__

#include "image_ppm.h"
#include <vector>

void relu(OCTET *ImgIn, int nH, int nW);
void convolution(OCTET *ImgIn, const std::vector<float> &filtre, OCTET *ImgOut,
                 unsigned int step, int nH, int nW);


class LayerConvPool { // Layer convolution + relu + pooling

  friend class CNN;
  friend class FullyConnectedLayer;

private:
  int nH, nW; // input size
  std::vector<OCTET *> input;
  std::vector<OCTET *> filtered_input;
  std::vector<OCTET *> output;

  std::vector<std::vector<float>> filtres;

public:
  LayerConvPool() {}

  void setup_layer(int nH, int nW,
                   const std::vector<std::vector<float>> &filtres) {
    this->nH = nH;
    this->nW = nW;
    this->filtres = filtres;
  }

  void conv() {
    filtered_input.resize(filtres.size() * input.size());
    for (size_t i = 0; i < filtres.size() * input.size(); i++) {
      allocation_tableau(filtered_input[i], OCTET, nH * nW);
    }
    size_t filtered_input_idx = 0;

    for (size_t i = 0; i < filtres.size(); i++) {
      for (size_t j = 0; j < input.size(); j++) {
        convolution(input[j], filtres[i], filtered_input[filtered_input_idx++],
                    1, nH, nW);
        relu(filtered_input[i + j], nH, nW);
      }
    }
  }

  void max_pooling() {
    output.resize(filtered_input.size());
    for (size_t i = 0; i < output.size(); ++i) {
      allocation_tableau(output[i], OCTET, nH * nW / 2);
    }

    OCTET max = 0;
    OCTET block[4];
    size_t output_idx = 0;

    for (size_t idx = 0; idx < output.size(); ++idx) {
      for (size_t i = 0; i < nH; i += 2) {
        for (size_t j = 0; j < nW; j += 2) {
          block[0] = filtered_input[idx][i * nH + j];
          block[1] = filtered_input[idx][i * nH + j + 1];
          block[2] = filtered_input[idx][((i + 1) * nH) + j];
          block[3] = filtered_input[idx][((i + 1) * nH) + j + 1];
          max = std::max(block[0],
                         std::max(block[1], std::max(block[2], block[3])));

          output[idx][output_idx++] = max;
        }
      }
      output_idx = 0;
    }
    /*char filename[250];
    for (int i = 0; i < output.size(); ++i) {
      sprintf(filename, "output_%d.pgm", i + 1);
      ecrire_image_pgm(filename, output[i], nH / 2, nW / 2);
    }*/
  }

  void process() {
    conv();
    max_pooling();
  }

  void set_input(std::vector<OCTET *> input) { this->input = input; }
};


#endif