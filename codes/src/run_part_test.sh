#!/bin/bash

python3 parallel_seq.py -i benchmark/HWMCC_15_17/6s160.aig > logs/log_parallel_6s160.txt
python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s160.aig > logs/log_part_test_6s160.txt

#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s343b31.aig > logs/log_part_test_6s343b31.txt 
#python3 parallel_seq.py -i benchmark/HWMCC_15_17/6s343b31.aig > logs/log_parallel_6s343b31.txt

#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s366r.aig > logs/log_part_test_6s366r.txt 
#python3 parallel_seq.py -i benchmark/HWMCC_15_17/6s366r.aig > logs/log_parallel_6s366r.txt

#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s191.aig > logs/log_part_test_6s191.txt 
#python3 parallel_seq.py -i benchmark/HWMCC_15_17/6s191.aig > logs/log_parallel_6s191.txt

#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s7.aig > logs/log_part_test_6s7.txt 
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s7.aig > results/new_log_tm_pred_6s7.txt
#
# python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s160.aig > logs/log_part_test_6s160.txt
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s160.aig > results/new_log_tm_pred_6s160.txt
##
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s366r.aig > results/new_log_tm_pred_6s366r.txt
#&&
#python3 partition_run_test.py -i benchmark/HWMCC_15_17/6s130.aig > logs/log_part_test_6s130.txt &&
#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s7.aig > logs/log_part_test_6s7.txt 
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s7.aig > results/new_log_tm_pred_6s7.txt
#python3 partition_run_test.py -i benchmark/HWMCC_15_17/6s20.aig > logs/log_part_test_6s20.txt 
#&&
#python3 partition_run_test_nr.py -i benchmark/HWMCC_15_17/6s191.aig > logs/log_part_test_6s191.txt 
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s191.aig > results/new_log_tm_pred_6s191.txt
# &&
#python3 partition_run_test.py -i benchmark/HWMCC_15_17/6s119.aig > logs/log_part_test_6s119.txt &&
# python3 partition_run_test.py -i benchmark/HWMCC_15_17/6s130.aig > logs/log_part_test_6s130.txt &&
# python3 partition_run_test.py -i benchmark/HWMCC_15_17/6s160.aig > logs/log_part_test_6s160.txt &&
#python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s3434b31.aig > results/new_log_tm_pred_6s3434b31.txt
#&&
# 
