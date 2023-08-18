CUDA_VISIBLE_DEVICES=3 python main.py \
    -dataset multiwoz_2.2 \
    -run_type train \
    -add_additional_loss \
    -model_dir mwoz_22_few_shot_0.5 \
    -epochs 20 \
    -batch_size 8\
    -warmup_ratio 0.1\
    -max_to_keep_ckpt 15 \
    -train_data_ratio 0.5 \
