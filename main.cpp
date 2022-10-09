#include "image_ppm.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "CNN.hpp"

using namespace std;

void softmax(double *vec, size_t size) {
  double exp_sum = 0;
  for (size_t i = 0; i < size; ++i) {
    exp_sum += exp(vec[i]);
  }

  for (size_t i = 0; i < size; ++i) {
    vec[i] = exp(vec[i]) / exp_sum;
  }
}

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
  char folder[100], folder2[100], image_to_predict[100];
  int nH, nW, taille;
  std::vector<OCTET *> image_set;

  if (argc != 4) {
    printf("Usage: folder_class1 folder_class2 image_to_predict\n");
    exit(EXIT_FAILURE);
  }

  sscanf(argv[1], "%s", folder);
  sscanf(argv[2], "%s", folder2);
  sscanf(argv[3], "%s", image_to_predict);

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

  std::vector<float> filtre = {0, -1, 0, -1, 5, -1, 0, -1, 0};

  std::vector<float> filtre2 = {0.5, 0.5, 0.5, -0.5, 1, 0.5, -0.5, -0.5, -0.5};

  std::vector<float> filtre3 = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> filtre4 = {0.5, 0, -0.5, 1, 0, -1, 0.5, 0, -0.5};

  std::vector<float> filtre5 = {0.5, 1, 0.5, 0, 0, 0, -0.5, -1, -0.5};

  CNN cnn(2, 5, nW, nH, image_set);
  cnn.addFilter(filtre);
  cnn.addFilter(filtre2);
  cnn.addFilter(filtre3);
  cnn.addFilter(filtre4);
  cnn.addFilter(filtre5);
  cnn.print_filters();

  cnn.setup_cnn();
  cnn.train();

  OCTET *ImgIn;

  lire_nb_lignes_colonnes_image_pgm(image_to_predict, &nH, &nW);
  taille = nH * nW;

  allocation_tableau(ImgIn, OCTET, taille);
  lire_image_pgm(image_to_predict, ImgIn, taille);

  cnn.predict(ImgIn, nH, nW);

  cnn.predict_class2_test();

  free(ImgIn);
  return EXIT_SUCCESS;
}
