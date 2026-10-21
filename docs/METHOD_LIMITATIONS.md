# Pocket quality and recommended reading

**Interrogating contrastive learning embeddings for structure-based virtual
screening: a case study on DrugCLIP**, Javier S. Utgés, David T. Jones and
Christine Orengo, bioRxiv preprint (2026).

- [Preprint](https://www.biorxiv.org/content/10.64898/2026.09.16.752029v1)
- [Analysis code](https://github.com/JavierSanchez-Utges/DrugCLIP-pocket-analysis)
- [Analysis data](https://zenodo.org/records/22791467)

The study examines pocket and ligand embeddings, pocket similarity, generalization
after removing training overlap, and screening with apo/holo structures,
AlphaFold models and predicted pockets. These comparisons help assess how pocket
definition and structure source affect retrieval performance.

## Comparing screening results

Record the structure source, pocket residues, candidate and reference libraries,
model weights, fold count and preprocessing settings. Keep these conditions fixed
when comparing code versions. Report retrieval metrics with their benchmark
definitions and evaluate experimental hit rates separately.

Name the background set used in each evaluation. For example, a ChemDiv sample
is a random-library background, while the standard DUD-E protocol uses its
provided decoys. Record how training/test overlap was handled alongside the results.
