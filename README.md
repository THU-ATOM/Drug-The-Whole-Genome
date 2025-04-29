# DrugCLIP for Drug-The-Whole-Genome

## Model weights and encoded embeddings

link: https://huggingface.co/datasets/bgao95/DrugCLIP_data

download model_weights.zip, encoded_mol_embs.zip, targets.zip, unzip them and put them inside ./data dir

## Set environment

you can set the environment with the Dockerfile in docker dir, or use the requirements.txt file.

## Do virtual screening

```
bash retrieval.sh
```

You need to set pocket path to ./data/targets/{target}/pocket.lmdb

target is one of the name in ./data/targets

you need to set num_folds to 8 for 5HT2A

The molecule library for the virtual screening is 1648137 molecules inside ChemDIV.

each line in result file look like this:

```
smiles,score
```

### For data won't fit in mem

To perform screening on chucked mol embedding files, run:

```shell
python utils/screening_chucks.py --gpu_num 8 --mol_embs <path_to_multiple_chucks> --zscore_embs <uniform_small_set_for_approx_zscore> --pocket_reps <path_to_pocket_reps_pkl> --batch_size <batch_size> --output_dir <path_to_output_dir> --rm_intermediate
```

The `--mol_embs` argument allows multiple args. Each arg should be a path to a chunk of molecule embeddings. The `--zscore_embs` argument should be a path to a small set of molecule embeddings for approximating the zscore; if not specified, the first `mol_embs` file is used. The resulting files will be saved as `merge{mol_embs_file_id}_{gpu_id}.pkl` in `output_dir` . The `--rm_intermediate` flag will remove the intermediate files after the next chunk for saving disk space.

To retrieve SMILES strings and original ids for the output files, run:

```shell
python utils/retrieve_chunk.py --input_files output/merge*.pkl --mol_lmdb <path_to_mol_lmdb> --output_dir retrieval_results --num_threads <num_threads>
```

The `--mol_lmdb` should be extractly the same order as the `--mol_embs` argument in the last step. The resulting files will be saved as `retrieval_results/{pocket_name}.csv`.

### Benchmarking throughput

```shell
time python utils/screening_benchmark_throughput.py --gpu_id 0 --mol_embs ZINC_0.npy --pocket_reps dtwg_af_embeddings.npy dtwg_af_names_.npy --output_dir ./benchmark_throughput/ --zscore_embs zscore_embs.npy
```

## Benchmarking

link: https://huggingface.co/datasets/bgao95/DrugCLIP_data

download DUD-E.zip, LIT-PCBA.zip, unzip them and put inside ./data dir

```
bash test.sh
```

select TASK to DUDE or PCBA in test.sh

## Other tools

Pocket Pretraining: https://github.com/THU-ATOM/ProFSA

virtual screening post-processing: https://github.com/THU-ATOM/DrugCLIP_screen_pipeline

Pocket detection: https://github.com/THU-ATOM/Pocket-Detection-of-DTWG
