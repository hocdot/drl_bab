# DRL-based Decision Heuristic for DNN Verification

### Pretrain GNN Embedding

```bash
python3 -m prepare.generate_raw_instances --dataset dnnv --n_vnnlib_per_onnx 100 --clean
python3 -m prepare.prepare_io --dataset dnnv --use_attack
python3 -m prepare.prepare_gnn_embedding --dataset dnnv
```

### Train

```bash
python3 train.py --mode train --dataset milp --n_input 3 --batch_size 10000 --actor_lr 0.001 --q_lr 0.001 --n_start_learning 100 --save_csv
python3 train.py --mode train --dataset dnnv --n_input 4 --batch_size 256 --n_train_episode 624 --n_start_learning 5000 --save_csv --save_offline_data
```

### Train Offline

```bash
python3 train.py --mode train_offline --dataset dnnv --n_input 4
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

- [ ] Constrastive loss
- [x] Graph embedding multiple networks
- [x] Minmax scaler 

- Ablation:
- [ ] Action weight * FSB score
- [ ] Train no embedding