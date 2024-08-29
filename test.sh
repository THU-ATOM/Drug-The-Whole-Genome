results_path="./test"  # replace to your results path
batch_size=64
weight_path="/drug/save_dir/journal/2024-08-07_17-17-03/checkpoint_best.pt"
#weight_path="/drug/save_dir/journal/2024-08-07_14-59-16/checkpoint_best.pt"
weight_path="/drug/save_dir/journal/2024-08-07_14-58-34/checkpoint_best.pt"

weight_path="/drug/save_dir/journal/2024-08-07_19-30-48/checkpoint_best.pt"
weight_path="/drug/save_dir/journal/2024-08-07_20-35-37/checkpoint_best.pt"


weight_path="/drug/save_dir/journal/2024-08-08_16-59-04/checkpoint_best.pt"
weight_path="/drug/save_dir/journal/2024-08-08_16-59-36/checkpoint_best.pt"


weight_path="/drug/save_dir/journal/2024-08-08_20-23-38/checkpoint_best.pt"
weight_path="/drug/save_dir/journal/2024-08-08_20-23-51/checkpoint_best.pt"

weight_path="/drug/save_dir/journal/2024-08-09_15-39-39/checkpoint_best.pt"
weight_path="/data/protein/save_dir/affinity/2023-05-06_22-08-56/checkpoint_best.pt"

TASK="DUDE" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="2" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 256 \
       --test-task $TASK \