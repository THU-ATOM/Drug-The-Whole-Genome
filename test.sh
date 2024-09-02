results_path="./test"  # replace to your results path
batch_size=64

weight_path="./data/benchmark_weights/90.pt"
use_folds=True

TASK="PCBA" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="1" python ./unimol/test.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --use-folds $use_folds \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \