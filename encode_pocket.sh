
echo "First argument: $1"

device=$1

batch_size=8
pocket_dir=$2 # path to the pocket dir





CUDA_VISIBLE_DEVICES=$device python ./unimol/encode_pockets.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --num-workers 1 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --seed 1 \
       --log-interval 100 --log-format simple \
       --pocket-dir $pocket_dir 
