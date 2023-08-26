#!/bin/bash

python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s343b31.aig > results/new_log_tm_pred_6s343b31.txt  
python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s160.aig > results/new_log_tm_pred_6s160.txt  
python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s366r.aig > results/new_log_tm_pred_6s366r.txt  
python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s119.aig > results/new_log_tm_pred_6s119.txt          
python3 MABMC_to_predict.py -i benchmark/HWMCC_15_17/6s341r.aig > results/new_log_tm_pred_6s341r.txt 
