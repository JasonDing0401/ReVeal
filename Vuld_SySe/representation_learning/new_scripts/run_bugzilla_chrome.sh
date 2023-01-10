title="bugzilla_chrome"
python -u api_test.py --dataset bugzilla_snykio_V3 --testset chrome_debian/bugzilla_snykio_V3 \
    --num_layers 1 --max_patience 25 \
    --pretrain bugzilla_snykio_V3_2022-11-29_23-30-15_f1.bin |& tee -a logs/${title}_$(date "+%m.%d-%H.%M.%S").log