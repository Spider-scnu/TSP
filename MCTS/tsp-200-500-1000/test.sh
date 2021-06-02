#!/usr/bin/env sh
rm -r kernel.o 
python ./utils/preprocess.py
nvcc -L /usr/local/cuda-9.0/lib64/ -I /usr/local/cuda-9.0/samples/common/inc   -lcudart -lcuda  -c kernel.cu


rm -r TSP.o
rm -r test
clear

python ./utils/preprocess2.py

g++ -lstdc++ -std=c++11 -fPIC -fvisibility=hidden -o  TSP.o -c TSP.cpp  -I/usr/local/cuda-9.0/samples/common/inc

g++ -lstdc++ -std=c++11 -fPIC -o test TSP.o kernel.o -L. -L/usr/local/cuda-9.0/lib64/ -lcudart -lcuda -lpthread -lm -ldl

threads=(4)
GPUNum=(8)
CityNum=(20)
for ((j=0;j<$GPUNum;j++));do
{
for ((i=0;i<$threads;i++));do
{
	
	./test $[$[$threads * $j] + $i] ../results/${CityNum}/results_tsp${CityNum}_batch_$[$[$threads * $j] + $i].txt ../instances/tsp${CityNum}_test_concorde.txt ${CityNum} $j

}&
done
wait
}&
done
wait

echo "Done."
