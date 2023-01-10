title="chrome_cwe20"
python -u api_test.py --dataset chrome_debian --testset CWE-20/chrome_debian \
    --num_layers 1 --max_patience 25 \
    --pretrain chrome_debian_2022-11-21_01-38-00.bin |& tee -a logs/${title}_$(date "+%m.%d-%H.%M.%S").log