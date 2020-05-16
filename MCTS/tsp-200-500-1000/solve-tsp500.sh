#!/usr/bin/env sh
rm -r TSP.o
rm -r test
g++ -lstdc++ -std=c++11 -fPIC -fvisibility=hidden -o  TSP.o -c TSP.cpp  -I/usr/local/cuda-9.0/samples/common/inc

g++ -lstdc++ -std=c++11 -fPIC -o test TSP.o kernel.o -L. -L/usr/local/cuda-9.0/lib64/ -lcudart -lcuda -lpthread -lm -ldl


./test ./results/result_500.txt ./tsp500_test_concorde.txt 500

echo "Done."
