# TA&AT

This is code for the paper "TA&AT: Enhancing Task-Oriented Dialog with Turn-Level Auxiliary Tasks and Action-Tree Based Scheduled Sampling".

First download the preprocessed datasets and TreeBase for each MultiWOZ dataset from our provided google drive links. (https://drive.google.com/file/d/1wBbZKYpweCKbwVHIJLV3DxcJh-KX_UXc/view?usp=sharing), unzip it in this directory.

## Environment setting

Our python version is 3.8.8.

The package can be installed by running the following command.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training

Our implementation supports a single GPU. Please use smaller batch size and more accumulation steps to ensure the toal batch size if out-of-memory error raises.

```
sh scripts/train.sh
```

## Inference

```
sh scripts/test.sh
```

All checkpoints are saved in ```$MODEL_DIR``` with names such as 'ckpt-epoch10'.

The result file (```$MODEL_OUTPUT```) will be saved in the checkpoint directory.


## Standardized Evaluation

For the MultiWOZ benchmark, we recommend to use [standardized evaluation script](https://github.com/Tomiinek/MultiWOZ_Evaluation).

```
# MultiWOZ2.2 is used for the benchmark (MultiWOZ2.2 should be preprocessed prior to this step)
python main.py -run_type predict -ckpt $CHECKPOINT -output $MODEL_OUTPUT -batch_size $BATCH_SIZE -version 2.2
# convert format for the the standardized evaluation, before running it, you should first change the variable 'file_path' to your raw inference file path, and change the variable 'file_path_1' to your target output file path.

python convert_to_standard.py

# clone the standardized evaluation repository
git clone https://github.com/Tomiinek/MultiWOZ_Evaluation
cd MultiWOZ_Evaluation
pip install -r requirements.txt

# do standardized evaluation
python evaluate.py -i $CONVERTED_MODEL_OUTPUT -b -s -r
```

## Acknowledgements

This code is based on the released code (https://github.com/bepoetree/MTTOD) for "Improving End-to-End Task-Oriented Dialogue System with A Simple Auxiliary Task" which distributed under Apache License Version 2.0. 
Copyright 2021- Yohan Lee. 

For the pre-trained language model, we use huggingface's Transformer (https://huggingface.co/transformers/index.html#), which distributed under Apache License Version 2.0. 
Copyright 2018- The Hugging Face team. All rights reserved.
