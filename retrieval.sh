
echo "First argument: $1"

MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="pocket.lmdb" # path to the pocket file
POCKET_PATH="./data/targets/5HT2A/pocket.lmdb"
#POCKET_PATH="/data/protein/DrugClip/Validation_data/5HT2A/PDB/pocket.lmdb"
num_folds=8
use_cache=True
save_path="5ht2a_new.txt"




CUDA_VISIBLE_DEVICES="1" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --num-folds $num_folds \
       --use-cache $use_cache \
       --save-path $save_path