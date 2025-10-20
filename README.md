# DRL-based Decision Heuristic for DNN Verification

### Pretrain GNN Embedding

```bash
python3 -m prepare.generate_raw_instances --dataset dnnv --n_vnnlib_per_onnx 100 --clean
python3 -m prepare.prepare_io --dataset dnnv --use_attack
python3 -m prepare.prepare_gnn_embedding --dataset dnnv
```

### Create offline data

```bash
python3 train.py --mode train --config condig/dnnv_create_offline_data
```

### Train Offline

```bash
python3 train.py --mode train_offline --config condig/dnnv.yaml
```

### Train

```bash
python3 train.py --mode train --config config/milp_10_10.yaml
python3 train.py --mode train --config config/milp_10_10_scale.yaml
```

```bash
python3 train.py --mode train --config condig/dnnv.yaml
python3 train.py --mode train --config condig/dnnv_debug.yaml
```


### Test

```bash
python3 infer.py --config config/milp_10_10.yaml
python3 infer.py --config config/milp_10_10_scale.yaml
python3 infer.py --config condig/dnnv.yaml
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