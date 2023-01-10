title="combined_combined"
python -u api_test.py --dataset combined --testset combined \
    --num_layers 1 --max_patience -1 --pretrain combined_2022-11-30_00-46-25_acc.bin |& tee -a logs/${title}_$(date "+%m.%d-%H.%M.%S").log