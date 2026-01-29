# DRL-based Decision Heuristic for DNN Verification

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Pretrain GNN Embedding

- Generate random instances from existing ONNX at `data/dataset/dnnv/onnx`:
    ```bash
    python3 -m prepare.generate_raw_instances --dataset dnnv --n_vnnlib_per_onnx 5000 --clean
    ```

- Filter out easy SAT instances:
    ```bash
    python3 -m prepare.prepare_io --dataset dnnv --use_attack
    ```

- Filter out easy UNSAT instances:
    ```bash
    python3 -m prepare.prepare_hard_instances --dataset dnnv
    ```

- Train GCN and extract embeddings:
    ```bash
    python3 -m prepare.prepare_gnn_embedding --dataset dnnv --gcn_embed_batch_size 32
    ```

- Split train and test sets:
    ```bash
    python3 -m prepare.split_train_test --dataset dnnv
    ```

### 3. Train RSB heuristic

```bash
python3 train.py --mode train --config config/dnnv.yaml
```


### 4. Test RSB/FSB/Polarity/Random heuristics

```bash
python3 infer.py --config config/dnnv.yaml
```
