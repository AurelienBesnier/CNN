#include "image_ppm.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

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

class CNN {

private:
  unsigned int n; // couches
  unsigned int h; // filtres

  int nH, nW; // input size

  std::vector<LayerConvPool> layers;
  std::vector<OCTET *> inputs;
  std::vector<OCTET *> class1;
  std::vector<OCTET *> class2;

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
    
    for (size_t i = inputs.size()/2; i < inputs.size(); ++i) {
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
    for (i = 0; i < n; ++i) {
      if (i == 0)
        layers[i].set_input(class1);
      else
        layers[i].set_input(layers[i - 1].output);
      layers[i].process();
    }

    outputLayer.set_input(layers[i-1].output);
    outputLayer.vectorise("class1_vector.pgm");
    
    i = 0;
    for (i = 0; i < n; ++i) {
      if (i == 0)
        layers[i].set_input(class2);
      else
        layers[i].set_input(layers[i - 1].output);
      layers[i].process();
    }

    outputLayer.set_input(layers[i-1].output);
    outputLayer.vectorise("class2_vector.pgm");
  }

  void predict(OCTET *img, int nH, int nW);
};

void relu(OCTET *ImgIn, int nH, int nW) {
  size_t taille = nH * nW;
  for (size_t i = 0; i < taille; i++) {
    if (ImgIn[i] < 128) {
      ImgIn[i] = 128;
    }
  }
}

void convolution(OCTET *ImgIn, const std::vector<float> &filtre, OCTET *ImgOut,
                 unsigned int step, int nH, int nW) {
  size_t taille = nH * nW;
  int filtre_idx = 0;

  for (size_t x = 0; x < nW; x++) {
    for (size_t y = 0; y < nH; y++) {
      for (size_t i = x - 1; i <= x + 1; i++) {
        for (size_t j = y - 1; j <= y + 1; j++) {
          if ((i > 0 && j > 0) && (i < nW && j < nH)) {
            double tmp = ImgIn[(i * nW + j)] * filtre[filtre_idx++] / 9;
            ImgOut[(x * nW + y)] += tmp;
            if (filtre_idx >= 9)
              filtre_idx = 0;
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  char folder[100], folder2[100];
  int nH, nW, taille;
  std::vector<OCTET *> image_set;

  if (argc != 3) {
    printf("Usage: folder_class1 folder_class2 \n");
    exit(EXIT_FAILURE);
  }

  sscanf(argv[1], "%s", folder);
  sscanf(argv[2], "%s", folder2);

  string path = folder;
  string path2 = folder2;
  std::vector<string> db1, db2;

  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    db1.push_back(entry.path());
  }

  for (const auto &entry : std::filesystem::directory_iterator(path2)) {
    db2.push_back(entry.path());
  }

  /*for(size_t i =0; i< db1.size(); ++i)
    cout<<db1[i]<<endl;*/

  size_t set_idx = 0;
  image_set.resize(db1.size() + db2.size());
  for (size_t i = 0; i < db1.size(); i++) {
    lire_nb_lignes_colonnes_image_pgm(const_cast<char *>(db1[i].c_str()), &nH,
                                      &nW);
    taille = nH * nW;
    allocation_tableau(image_set[set_idx++], OCTET, taille);
    lire_image_pgm(const_cast<char *>(db1[i].c_str()), image_set[i], taille);
  }

  for (size_t i = 0; i < db2.size(); i++) {
    lire_nb_lignes_colonnes_image_pgm(const_cast<char *>(db2[i].c_str()), &nH,
                                      &nW);
    taille = nH * nW;
    allocation_tableau(image_set[set_idx++], OCTET, taille);
    lire_image_pgm(const_cast<char *>(db2[i].c_str()), image_set[i], taille);
  }

  /*OCTET *ImgIn;

  lire_nb_lignes_colonnes_image_pgm(folder, &nH, &nW);
  taille = nH * nW;

  allocation_tableau(ImgIn, OCTET, taille);
  lire_image_pgm(folder, ImgIn, taille);*/

  std::vector<float> filtre = {0, -1, 0, -1, 4, -1, 0, -1, 0};

  std::vector<float> filtre2 = {0, 1, 0, 1, -4, 1, 0, 1, 0};

  std::vector<float> filtre3 = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> filtre4 = {0.5, 0, -0.5, 1, 0, -1, 0.5, 0, -0.5};

  std::vector<float> filtre5 = {0.5, 1, 0.5, 0, 0, 0, -0.5, -1, -0.5};

  // image_set.push_back(ImgIn);
  CNN cnn(2, 5, nW, nH, image_set);
  cnn.addFilter(filtre);
  cnn.addFilter(filtre2);
  cnn.addFilter(filtre3);
  cnn.addFilter(filtre4);
  cnn.addFilter(filtre5);
  cnn.print_filters();

  cnn.setup_cnn();
  cnn.train();

  // free(ImgIn);
  return EXIT_SUCCESS;
}
