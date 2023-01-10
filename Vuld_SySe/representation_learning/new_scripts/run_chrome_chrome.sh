title="chrome_chrome"
python -u api_test.py --dataset chrome_debian --testset chrome_debian \
    --num_layers 1 --max_patience -1 --pretrain chrome_debian_2022-11-29_23-26-55_f1.bin |& tee -a logs/${title}_$(date "+%m.%d-%H.%M.%S").log