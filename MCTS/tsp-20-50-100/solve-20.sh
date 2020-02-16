#!/bin/bash
# author: 
rm -r ./code/TSP.o
rm -r ./test
make

Total_Instance_Num=(10000)
Temp_City_Num=(20)
threads=$1
Inst_Num_Per_Batch=$((1+$Total_Instance_Num/$threads))
tsp=("./tsp20_test_concorde.txt")
j=0
sources20=("./results/20/result_1.txt" "./results/20/result_2.txt" "./results/20/result_3.txt" "./results/20/result_4.txt" "./results/20/result_5.txt" "./results/20/result_6.txt" "./results/20/result_7.txt" "./results/20/result_8.txt" "./results/20/result_9.txt" "./results/20/result_10.txt" "./results/20/result_11.txt" "./results/20/result_12.txt" "./results/20/result_13.txt" "./results/20/result_14.txt" "./results/20/result_15.txt" "./results/20/result_16.txt" "./results/20/result_17.txt" "./results/20/result_18.txt" "./results/20/result_19.txt" "./results/20/result_20.txt" "./results/20/result_21.txt" "./results/20/result_22.txt" "./results/20/result_23.txt" "./results/20/result_24.txt" "./results/20/result_25.txt" "./results/20/result_26.txt" "./results/20/result_27.txt" "./results/20/result_28.txt" "./results/20/result_29.txt" "./results/20/result_30.txt" "./results/20/result_31.txt" "./results/20/result_32.txt")
for ((i=0;i<$threads;i++));do
{
	./test $i ${sources20[i]} ${tsp[j]} ${Temp_City_Num[j]} ${Inst_Num_Per_Batch}
}&
done
wait
echo "Done."