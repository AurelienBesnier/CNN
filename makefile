all:
	g++ -o cnn main.cpp FullyConnectedLayer.hpp LayerConvPool.hpp CNN.hpp  -lm -g

clean:
	rm cnn output_*
