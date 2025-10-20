# DRL-based Decision Heuristic for DNN Verification

### Pretrain GNN Embedding

```bash
python3 -m prepare.generate_raw_instances --dataset dnnv --n_vnnlib_per_onnx 100 --clean
python3 -m prepare.prepare_io --dataset dnnv --use_attack
python3 -m prepare.prepare_gnn_embedding --dataset dnnv
```

### Create offline data

```bash
python3 train.py --mode train --dataset dnnv --n_input 4 --batch_size 1000 --n_train_instance 624 --n_start_learning 8000 --n_start_save_offline_data 2000 --save_offline_data --n_train_epoch 3 --use_embedding
```

### Train Offline

```bash
python3 train.py --mode train_offline --dataset dnnv --n_input 4 --save_csv --use_embedding
```

### Train

```bash
python3 train.py --mode train --dataset milp --n_input 3 --batch_size 10000 --actor_lr 0.001 --q_lr 0.001 --n_start_learning 100 --save_csv --device cuda:0 --use_scale_action --checkpoint_dirname scale

python3 train.py --mode train --dataset milp --n_input 3 --batch_size 10000 --actor_lr 0.001 --q_lr 0.001 --n_start_learning 100 --device cuda:1

python3 train.py --mode train --dataset dnnv --n_input 4 --batch_size 1000 --n_train_instance 624 --n_start_learning 500 --save_csv --n_train_epoch 10 --n_qf_optimize 2 --use_embedding

```
```bash
python3 train.py --mode train --dataset dnnv --n_input 4 --batch_size 10 --n_train_instance 10 --n_start_learning 1 --save_csv --n_train_epoch 1 --n_qf_optimize 1 --checkpoint_dirname ckpt2 --use_embedding
```


### Test

```bash
python3 infer.py
python3 infer.py --dataset dnnv --n_input 4 --n_test_instance 9 --n_max_step 200 --n_max_remaining_subproblem 1000
python3 infer.py --dataset dnnv --n_input 4 --n_test_instance 50 --use_embedding
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