#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <algorithm>
#include "image_ppm.h"

void relu(OCTET*ImgIn, int nH, int nW);
void convolution(OCTET *ImgIn, const std::vector<float>  &filtre, OCTET* ImgOut, unsigned int step, int nH, int nW);

class LayerConvPool { // Layer convolution + relu + pooling

  friend class CNN;
  friend class FullyConnectedLayer;

  private:
    int nH, nW; // input size
    std::vector<OCTET*> input;
    std::vector<OCTET*> filtered_input;
    std::vector<OCTET*> output;


    std::vector<std::vector<float>> filtres;

  public:
    LayerConvPool()
    {}

    void setup_layer(int nH, int nW, const std::vector<std::vector<float>> & filtres)
    {
      this->nH = nH;
      this->nW = nW;
      this->filtres = filtres;
    }

    void conv()
    {
      filtered_input.resize(filtres.size()*input.size());
      for(size_t i = 0; i<filtres.size()*input.size(); i++)
      {
          allocation_tableau(filtered_input[i], OCTET, nH*nW);
      }
      size_t filtered_input_idx = 0;

      for(size_t i = 0; i<filtres.size(); i++)
      {
        for(size_t j = 0; j < input.size(); j++)
        {
          convolution(input[j], filtres[i], filtered_input[filtered_input_idx++],1,nH,nW);
          relu(filtered_input[i+j], nH, nW);
        }
      }

      char filename[250];
      for(size_t i = 0; i<filtered_input.size(); ++i)
      {
        sprintf(filename, "conv_%d.pgm",i+1);
        ecrire_image_pgm(filename, filtered_input[i],nH, nW);
      }

    }

    void max_pooling()
    {
      output.resize(filtered_input.size());
      for(size_t i = 0; i<output.size(); ++i)
      {
        allocation_tableau(output[i], OCTET, nH*nW/2);
      }

      OCTET max = 0;
      OCTET block[4];
      size_t output_idx = 0;

      for(size_t idx = 0; idx < output.size(); ++idx)
      {
        for(size_t i = 0; i<nH; i+=2)
        {
          for(size_t j = 0; j<nW; j+=2)
          {
            block[0] = filtered_input[idx][i*nH+j];
            block[1] = filtered_input[idx][i*nH+j+1];
            block[2] = filtered_input[idx][((i+1)*nH)+j];
            block[3] = filtered_input[idx][((i+1)*nH)+j+1];
            max = std::max(block[0], std::max(block[1], std::max(block[2], block[3])));

            output[idx][output_idx++]=max;
          }
        }
        output_idx = 0;

      }
      char filename[250];
      for(size_t i = 0; i<output.size(); ++i)
      {
        sprintf(filename, "output_%d.pgm",i+1);
        ecrire_image_pgm(filename, output[i],nH/2, nW/2);
      }
   
    }

    void process()
    {
      conv();
      max_pooling();
    }

    void set_input(std::vector<OCTET*> input) {this->input = input;}

};

class FullyConnectedLayer
{
  private:
    std::vector<OCTET*> input;
    OCTET* vectorised_input;
    int nH, nW;

  public:
    FullyConnectedLayer()
    {}

    void set_input(std::vector<OCTET*> &input) {this->input = input;}
    void setup_outputLayer(int nH, int nW) {this->nH=nH; this->nW=nW;}

    void vectorise()
    {
      allocation_tableau(vectorised_input, OCTET, nH*nW);
      printf("nH=%d; nW=%d\n",nH,nW);
    }
};


class CNN {

  private:
    unsigned int n; // couches
    unsigned int h; // filtres

    int nH, nW;     //input size

    std::vector<LayerConvPool> layers;
    std::vector<OCTET*> inputs;

    FullyConnectedLayer outputLayer;


  public:
    std::vector<std::vector<float>> filtres;

    CNN(unsigned int n, unsigned int h, int nH, int nW, std::vector<OCTET*> input) : n(n), h(h), nH(nH), nW(nW), inputs(input)
    {}

    void setup_cnn()
    {
      layers.resize(n);
      size_t i = 0;
      for(i = 0; i<layers.size(); i++)
      {
        layers[i].setup_layer(nH/(i+1),nW/(i+1),filtres);
      }
      outputLayer.setup_outputLayer(nH/(layers.size()*2), nW/(layers.size()*2));
    }

    void addFilter(std::vector<float>& filtre)
    {filtres.push_back(filtre);}

    void print_filters() const
    {
      for(size_t i = 0; i < h; i++)
      {
        printf("[ %f, %f, %f\n%f, %f, %f\n %f, %f, %f]\n",
        filtres[i][0],filtres[i][1],filtres[i][2],
        filtres[i][3],filtres[i][4],filtres[i][5],
        filtres[i][6],filtres[i][7],filtres[i][8]);
        printf("\n");
      }
    }

    void train()
    {
      size_t i=0;
      for(i = 0; i < n; ++i){
        if(i == 0)
          layers[i].set_input(inputs);
        else
          layers[i].set_input(layers[i-1].output);
        layers[i].process();
      }

      outputLayer.set_input(layers[i].output);
      outputLayer.vectorise();
    }

    void predict(OCTET* img, int nH, int nW);
};


void relu(OCTET*ImgIn, int nH, int nW)
{
  size_t taille = nH*nW;
  for(size_t i = 0; i< taille; i++)
  {
    if(ImgIn[i] < 128){
      ImgIn[i] = 128;
    }
  }
}

void convolution(OCTET *ImgIn, const std::vector<float>  &filtre, OCTET* ImgOut, unsigned int step, int nH, int nW)
{
  size_t taille = nH*nW;
  int filtre_idx = 0;
  
 for(size_t x = 0; x < nW; x++)
 {
    for(size_t y = 0; y < nH; y++)
    {
      for(size_t i = x - 1; i <= x+1; i++)
      {
        for(size_t j = y - 1; j <= y+1; j++)
        {
          if((i>0 && j>0) && (i<nW && j<nH)) 
          {
            double tmp = ImgIn[(i*nW+j)] * filtre[filtre_idx++] / 9;
            ImgOut[(x * nW+ y)] += tmp;
            if(filtre_idx >= 9)
              filtre_idx = 0;
          }
        } 
      }
    }
 }  
}

int main(int argc, char* argv[])
{
   char cNomImgLue[250],  cNomImgEcrite[250];
   int nH, nW, taille;
   std::vector<OCTET*> image_set;

  if (argc != 3)
     {
       printf("Usage: folder ImageAClasser.pgm\n");
       exit (1) ;
     }
   
   sscanf (argv[1],"%s",cNomImgLue);
   sscanf (argv[2],"%s",cNomImgEcrite);
   OCTET *ImgIn;
   
   lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
   taille = nH * nW;
  

   allocation_tableau(ImgIn, OCTET, taille);
   lire_image_pgm(cNomImgLue, ImgIn, taille);
   

   std::vector<float> filtre = {0,-1,0
                   ,-1,4,-1
                   ,0,-1,0};

  std::vector<float> filtre2 = {0,1,0
                   ,1,-4,1
                   ,0,1,0};

  std::vector<float> filtre4 = {0.5,0,-0.5
                   ,1,0,-1
                   ,0.5,0,-0.5};

  std::vector<float> filtre5 = {0.5,1,0.5
                   ,0,0,0
                   ,-0.5,-1,-0.5};

  image_set.push_back(ImgIn);
  CNN cnn(2,4,nW,nH,image_set);
  cnn.addFilter(filtre);
  cnn.addFilter(filtre2);
  cnn.addFilter(filtre4);
  cnn.addFilter(filtre5);
  cnn.print_filters();

  cnn.setup_cnn();
  cnn.train();

  free(ImgIn);
  return 1;
}
