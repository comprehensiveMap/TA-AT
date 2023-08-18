CUDA_VISIBLE_DEVICES=0 python main.py \
    -run_type predict \
    -batch_size 8\
    -ckpt multiwoz_2.0/ckpt-epoch7 \
    -output epoch7_inference \
    -dataset multiwoz2.0 \

