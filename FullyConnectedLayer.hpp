#ifndef __FULLYCONNECTEDLAYER_HPP__
#define __FULLYCONNECTEDLAYER_HPP__

#include "image_ppm.h"
#include <vector>

class FullyConnectedLayer {
private:
  std::vector<OCTET *> input;
  double* vect;
  OCTET *vectorized_input;
  int nH, nW;

public:
  FullyConnectedLayer() {}

  void set_input(std::vector<OCTET *> &input) { this->input = input; }
  void setup_outputLayer(int nH, int nW) {
    this->nH = nH;
    this->nW = nW;
  }

  void vectorise(char * vectorName) {
    size_t size = nH * nW;
    allocation_tableau(vectorized_input, OCTET, size);
    allocation_tableau(vect, double, size);
    printf("nH=%d; nW=%d\n", nH, nW);
    printf("input size = %d\n", input.size());

    for (size_t i = 0; i < input.size(); ++i) {
      for (size_t j = 0; j < size; ++j) {
        vect[j] += (double)input[i][j] /(double) input.size();
      }
    }
    
    for (size_t j = 0; j < size; ++j) {
        vectorized_input[j]=vect[j];
      }
    
    ecrire_image_pgm(vectorName,vectorized_input, nH, nW);
  }
};

#endif