# DrugCLIP_vs


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







