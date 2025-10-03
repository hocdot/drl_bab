# DRL-based Decision Heuristic for DNN Verification

### Pretrain GNN Embedding

```bash
python3 -m prepare.generate_raw_instances --dataset dnnv --vnnlib_eps 0.08 --n_vnnlib_per_onnx 10 --clean
python3 -m prepare.prepare_io --dataset dnnv
python3 -m prepare.prepare_gnn_embedding --dataset dnnv
```

### Train

```bash
python3 train.py
python3 train.py --dataset dnnv --n_input 4 --n_train_episode 50
```

### Test

```bash
python3 infer.py
```
