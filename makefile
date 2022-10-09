all:
	g++ -o cnn CNN.cpp -lm -g

clean:
	rm cnn output_*
