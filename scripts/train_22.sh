CUDA_VISIBLE_DEVICES=3 python main.py \
    -dataset multiwoz_2.2 \
    -run_type train \
    -with_tree \
    -mu 20 \
    -add_additional_loss \
    -model_dir multiwoz_2.2_with_tree_mu_20 \
    -epochs 10 \
    -batch_size 8\
    -warmup_ratio 0.1\
    -max_to_keep_ckpt 10 \