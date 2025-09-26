# DRL-based Decision Heuristic for DNN Verification

### Pretrain GNN Embedding

```bash
python3 -m prepare.prepare_io --dataset dnnv
python3 -m prepare.prepare_gnn_embedding --dataset dnnv
```

### Train

```bash
python3 train.py
```

### Test

```bash
python3 infer.py
```
