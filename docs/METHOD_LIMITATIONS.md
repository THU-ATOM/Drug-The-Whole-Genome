# Recommended reading and applicability

**Interrogating contrastive learning embeddings for structure-based virtual
screening: a case study on DrugCLIP**, Javier S. Utgés, David T. Jones and
Christine Orengo, bioRxiv preprint (2026).

- [Preprint](https://www.biorxiv.org/content/10.64898/2026.09.16.752029v1)
- [Analysis code](https://github.com/JavierSanchez-Utges/DrugCLIP-pocket-analysis)
- [Analysis data](https://zenodo.org/records/22791467)

The authors study pocket/ligand embedding geometry, pocket similarity, training
overlap removal, and sensitivity to pocket sources including apo/holo,
AlphaFold and predicted pockets. Their reported results support both useful
retrieval capability and limits under more realistic pocket inputs. See the
preprint for the exact benchmark definitions, conditions and numerical claims.

Use this work as context for interpreting performance. It does not establish
that every poor result is a user error, that every new target will work, or that
retrieval scores are experimental affinity measurements. Recovering a known
ligand in the top fraction of a candidate library is not the same quantity as
a prospective wet-lab hit rate.

When reporting results, state the structure/pocket source, candidate and reference
libraries, model/fold version, preprocessing, metrics and overlap policy. Do not
call random background molecules experimentally established inactives. A custom
ChemDiv-background evaluation should be named as such rather than equated with
the standard DUD-E decoy protocol.
