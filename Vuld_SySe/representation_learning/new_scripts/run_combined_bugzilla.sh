title="combined_bugzilla"
python -u api_test.py --dataset combined --testset chrome_debian/bugzilla_snykio_V3 \
    --num_layers 1 --max_patience 25 \
    --pretrain combined_2022-11-29_23-30-29_f1.bin |& tee -a logs/${title}_$(date "+%m.%d-%H.%M.%S").log