#ifndef __CNN_HPP__
#define __CNN_HPP__

#include <cmath>
#include <iostream>
#include <vector>

#include "FullyConnectedLayer.hpp"
#include "LayerConvPool.hpp"
#include "image_ppm.h"

class CNN {

private:
  unsigned int n; // couches
  unsigned int h; // filtres

  int nH, nW; // input size

  std::vector<LayerConvPool> layers;
  std::vector<OCTET *> inputs;
  std::vector<OCTET *> class1;
  std::vector<OCTET *> class2;

  double *vect_class1;
  double *vect_class2;

  FullyConnectedLayer outputLayer;

public:
  std::vector<std::vector<float>> filtres;

  CNN(unsigned int n, unsigned int h, int nH, int nW,
      std::vector<OCTET *> input)
      : n(n), h(h), nH(nH), nW(nW), inputs(input) {}

  void setup_cnn() {
    for (size_t i = 0; i < inputs.size() / 2; ++i) {
      class1.push_back(inputs[i]);
    }

    for (size_t i = inputs.size() / 2; i < inputs.size(); ++i) {
      class2.push_back(inputs[i]);
    }

    layers.resize(n);
    size_t i = 0;
    for (i = 0; i < layers.size(); i++) {
      layers[i].setup_layer(nH / (i + 1), nW / (i + 1), filtres);
    }
    outputLayer.setup_outputLayer(nH / (n * 2), nW / (n * 2));
  }

  void addFilter(std::vector<float> &filtre) { filtres.push_back(filtre); }

  void print_filters() const {
    for (size_t i = 0; i < h; i++) {
      printf("[ %f, %f, %f\n%f, %f, %f\n %f, %f, %f]\n", filtres[i][0],
             filtres[i][1], filtres[i][2], filtres[i][3], filtres[i][4],
             filtres[i][5], filtres[i][6], filtres[i][7], filtres[i][8]);
      printf("\n");
    }
  }

  void train() {
    size_t i = 0;
    std::cout << "training on class 1..." << std::endl;
    for (i = 0; i < n; ++i) {
      if (i == 0)
        layers[i].set_input(class1);
      else
        layers[i].set_input(layers[i - 1].output);
      layers[i].process();
    }

    std::cout << "vectorizing..." << std::endl;
    outputLayer.set_input(layers[i - 1].output);
    outputLayer.vectorise("class1_vector.pgm");
    this->vect_class1 = outputLayer.get_vector();
    std::cout << "training done !" << std::endl;

    std::cout << "training on class 2..." << std::endl;
    i = 0;
    for (i = 0; i < n; ++i) {
      if (i == 0)
        layers[i].set_input(class2);
      else
        layers[i].set_input(layers[i - 1].output);
      layers[i].process();
    }

    std::cout << "vectorizing..." << std::endl;
    outputLayer.set_input(layers[i - 1].output);
    outputLayer.vectorise("class2_vector.pgm");
    this->vect_class2 = outputLayer.get_vector();
    std::cout << "training done !" << std::endl;
  }

  void predict(OCTET *img, int nH, int nW) {
    std::vector<OCTET *> holder;
    holder.push_back(img);

    std::cout << "Prediction started!" << std::endl;
    size_t i = 0;
    for (i = 0; i < n; ++i) {
      if (i == 0)
        layers[i].set_input(holder);
      else
        layers[i].set_input(layers[i - 1].output);
      layers[i].process();
    }

    outputLayer.set_input(layers[i - 1].output);
    outputLayer.vectorise("img_to_predict_vector.pgm");

    double *vector_image = outputLayer.get_vector();

    // Compare with vector of the two classes
    double score_class1 = 0;
    double score_class2 = 0;

    for (size_t idx = 0; idx < ((nH / (n * 2)) * (nW / (n * 2))); idx++) {
      /*score_class1 +=
          vector_image[idx] * log10(vector_image[idx] / vect_class1[idx]);*/
	  score_class1 += sqrt(vector_image[idx]*vect_class1[idx]);
    }

    for (size_t idx = 0; idx < ((nH / (n * 2)) * (nW / (n * 2))); idx++) {
      /*score_class2 +=
          vector_image[idx] * log10(vector_image[idx] / vect_class2[idx]);*/
	score_class2 += sqrt(vector_image[idx]*vect_class2[idx]);
    }

    std::cout << "Score with class 1: " << score_class1 << std::endl;
    std::cout << "Score with class 2: " << score_class2 << std::endl;
  }
};

#endif