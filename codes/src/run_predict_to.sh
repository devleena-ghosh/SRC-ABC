#!/bin/bash

file='6s_seq_test.txt'
to=3600
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/${fn}.aig > results/new_log_tm_pred_${fn}.txt
#../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}_n.aig; print_stats; &get; bmc2 -S     0 -T   3600 -F 0 -v -L stdout; print_stats" > results/log_bmc2_3600_${fn}.txt 
#../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}_n.aig; print_stats; &get; bmc3 -g -S     0 -T   3600 -F 0 -v -L stdout; print_stats" > results/log_bmc3g_3600_${fn}.txt 
#(time (timeout ${to} ../super-prove-build/build2/super_prove/bin/super_deep.sh -r stdout ../benchmark/HWMCC_15_17/${fn}.aig) 2>&1) > logs/log_deep_${fn}_D2.txt 

#python3 MAB_BMC_frame.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_${fn}_D2_frame.txt

#if [["$i" -eq 4]]; then
#fi
#break
i=$((i+1))
done < ${file}
