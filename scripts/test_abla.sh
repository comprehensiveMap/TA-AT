CUDA_VISIBLE_DEVICES=3 python main.py \
    -run_type predict \
    -batch_size 8\
    -ckpt ablation_ta/ckpt-epoch9 \
    -output epoch9_inference \
    -dataset multiwoz_2.0 \

