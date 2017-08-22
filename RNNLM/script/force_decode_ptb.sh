PY=../python/run.py
MODEL_DIR=../model/model_ptb
TRAIN_PATH=../data/ptb/train
DEV_PATH=../data/ptb/valid
TEST_PATH=../data/ptb/test

export CUDA_VISIBLE_DEVICES=1


python $PY --mode FORCE_DECODE --model_dir $MODEL_DIR \
    --test_path $TEST_PATH \
    --size 20 --num_layers 2 \
    --n_epoch 100 --L 15 --n_bucket 3 --force_decode_output force_decode.txt
    
