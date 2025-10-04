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
python3 train.py --dataset dnnv --n_input 4 --n_train_episode 275 --batch_size 1000 --n_start_learning 2000
```

### Test

```bash
python3 infer.py
python3 infer.py --dataset dnnv --n_input 4 --n_test_episode 9 --n_max_step 200 --n_max_remaining_subproblem 1000
```

```bash
python3 -m dnnv_solver.main --net data/dataset/dnnv/onnx/mnist-net_256x2.onnx --spec data/dataset/dnnv/vnnlib/prop_1_0.03.vnnlib --disable_attack --disable_restart --disable_stabilize
```

### TODO

- [ ] Graph embedding + Constrastive loss
- [ ] Minmax scaler 

- Ablation:
- [ ] Action weight * FSB score
- [ ] Train no embedding