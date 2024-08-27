
echo "First argument: $1"

device=$1

results_path="./test"  # replace to your results path
batch_size=8
mol_path=$2 # path to the molecule file
save_path=$3 # path to the save dir





CUDA_VISIBLE_DEVICES=$device python ./unimol/encode_mols.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $mol_path \
       --save-dir $save_path 
