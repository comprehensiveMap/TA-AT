CUDA_VISIBLE_DEVICES=0 python main.py \
    -dataset multiwoz_2.1 \
    -run_type train \
    -add_additional_loss \
    -model_dir  multiwoz_2.1\
    -batch_size 8\
    -epochs 10\
    -warmup_ratio 0.1\
    -max_to_keep_ckpt 10 \
