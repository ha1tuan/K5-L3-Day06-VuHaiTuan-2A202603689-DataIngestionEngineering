# Day 06 retrieval summary

| Strategy | Split | N | Hit@1 | Evidence Hit@3 | Filter compliant | Mean top-1 score |
|---|---|---|---|---|---|---|
| A | dev | 18 | 8/18 (44%) | 11/18 (61%) | 2/2 | 0.8049 |
| A | test | 6 | 3/6 (50%) | 4/6 (67%) | 1/2 | 0.7766 |
| A | all | 24 | 11/24 (46%) | 15/24 (62%) | 3/4 | 0.7978 |
| B | dev | 18 | 10/18 (56%) | 12/18 (67%) | 2/2 | 0.7888 |
| B | test | 6 | 4/6 (67%) | 4/6 (67%) | 1/2 | 0.7655 |
| B | all | 24 | 14/24 (58%) | 16/24 (67%) | 3/4 | 0.7829 |
| C | dev | 18 | 10/18 (56%) | 12/18 (67%) | 2/2 | 0.7888 |
| C | test | 6 | 4/6 (67%) | 4/6 (67%) | 2/2 | 0.7655 |
| C | all | 24 | 14/24 (58%) | 16/24 (67%) | 4/4 | 0.7829 |

## Chunks

| Index | Chunks | Median chars | Approx tokens | Embedded this run | Ingest s |
|---|---|---|---|---|---|
| A | 1162 | 373.5 | 97875 | 0 | 0.25 |
| B | 556 | 759.0 | 97761 | 0 | 0.14 |

C uses the same chunks and vectors as B (index `B`) and only adds the metadata filter when the question has one; questions without a filter run unfiltered under C.
Hit@3 is an automatic proxy; coherence and grounding still need the manual sheets.
