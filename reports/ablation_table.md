# RAG Ablation Results

Each row toggles one or more pipeline components. `QE` = UMLS query expansion, `RR@k` = cross-encoder reranker keeping top-k, `dense@k` = dense-retrieval only, no reranker.

| Configuration | n | EM | F1 | BERTScore | ROUGE-L | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw-query + dense@1 | 60 | 0.0833 | 0.1009 | 0.8178 | 0.1042 | 0.083 | 0.000 | 0.083 |
| raw-query + dense@3 | 60 | 0.0833 | 0.1009 | 0.8178 | 0.1042 | 0.083 | 0.083 | 0.083 |
| raw-query + dense@5 | 60 | 0.0833 | 0.1009 | 0.8178 | 0.1042 | 0.083 | 0.083 | 0.083 |
| raw-query + rerank@1 | 60 | 0.0667 | 0.0849 | 0.8211 | 0.0854 | 0.117 | 0.000 | 0.117 |
| raw-query + rerank@3 | 60 | 0.0833 | 0.0978 | 0.8167 | 0.1007 | 0.117 | 0.167 | 0.133 |
| raw-query + rerank@5 | 60 | 0.0833 | 0.1033 | 0.8205 | 0.1059 | 0.117 | 0.167 | 0.137 |
| query-exp + dense@1 | 60 | 0.0333 | 0.0512 | 0.8140 | 0.0589 | 0.083 | 0.000 | 0.083 |
| query-exp + dense@3 | 60 | 0.0333 | 0.0512 | 0.8140 | 0.0589 | 0.083 | 0.083 | 0.083 |
| query-exp + dense@5 | 60 | 0.0333 | 0.0512 | 0.8140 | 0.0589 | 0.083 | 0.083 | 0.083 |
| query-exp + rerank@1 | 60 | 0.0500 | 0.0710 | 0.8179 | 0.0749 | 0.117 | 0.000 | 0.117 |
| query-exp + rerank@3 | 60 | 0.0667 | 0.0885 | 0.8152 | 0.0879 | 0.117 | 0.167 | 0.136 |
| query-exp + rerank@5 | 60 | 0.0667 | 0.0875 | 0.8170 | 0.0878 | 0.117 | 0.167 | 0.136 |
