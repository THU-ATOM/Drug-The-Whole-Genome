# DrugCLIP for Drug-The-Whole-Genome

## License

This project uses different licenses for different components:

- **Source Code**: Licensed under [Apache 2.0](LICENSE)
  - The source code is freely available for both academic and commercial use.

- **Database**: Licensed under [CC BY 4.0](docs/LICENSE.md)
  - The Drug-The-Whole-Genome database is freely available for both academic and commercial use with attribution.

- **Model Weights & Outputs**: Licensed under [CC BY-NC 4.0](docs/MODEL_WEIGHTS_LICENSE.md)
  - The DrugCLIP model weights and all results generated using the model are available for non-commercial use only.
  - For commercial use, please contact the authors for licensing options.

- **MIT Licensed Components**: [MIT License](unimol/LICENSE)
  - Part of the code is modified from Uni-Mol.
  - Copyright (c) 2022 DP Technology

Please see [LICENSE.md](docs/LICENSE.md) and [MODEL_WEIGHTS_LICENSE.md](docs/MODEL_WEIGHTS_LICENSE.md) for full details.

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

## Benchmarking

link: https://huggingface.co/datasets/bgao95/DrugCLIP_data

download DUD-E.zip, LIT-PCBA.zip, unzip them and put inside ./data dir


```
bash test.sh
```

select TASK to DUDE or PCBA in test.sh


## Understanding DrugCLIP — recommended reading

For an independent analysis of what DrugCLIP can and cannot do, see:

> **Interrogating contrastive learning embeddings for structure-based virtual screening: a case study on DrugCLIP**
> Javier S. Utgés, David T. Jones, Christine Orengo (UCL, Institute of Structural and Molecular Biology), bioRxiv 2026.
> https://www.biorxiv.org/content/10.64898/2026.09.16.752029v1 — DOI: 10.64898/2026.09.16.752029

The study systematically dissects what DrugCLIP's learned embeddings encode and examines its **capabilities**, **applicable scenarios**, and **generalization**. It may help answer questions about our work. In particular, the paper reports:

- **Generalization (de-leakage benchmark).** DrugCLIP generalizes to unseen proteins and chemistries rather than memorizing training data — recovering the bound ligand within the top 1% of 50,000 candidates for 55–75% of novel targets.
- **Pocket embeddings.** Structurally coherent and robust to conformational variation; similarity tracks geometric resemblance, and (though not trained for it) they reach state-of-the-art pocket-similarity search at >100× the speed of conventional descriptors.
- **Ligand embeddings.** Capture a pocket-aware, binding-context notion of chemical similarity — not just Tanimoto fingerprint identity.
- **Conformer robustness.** Embeddings stay consistent across alternative conformers of the same pocket/ligand.
- **Boundaries of applicability.** Performance depends on input pocket accuracy, decreasing as the structure deviates from holo — across holo, apo, AlphaFold, and predicted pockets — due to sidechain reorientation and (for predicted pockets) residue mismatch. Some enrichment ability remains even in the hardest setting.

If you have questions about our work, this paper's analysis is a good place to look.

## Other tools

Pocket Pretraining: https://github.com/THU-ATOM/ProFSA

virtual screening post-processing: https://github.com/THU-ATOM/DrugCLIP_screen_pipeline

Pocket detection: https://github.com/THU-ATOM/Pocket-Detection-of-DTWG







