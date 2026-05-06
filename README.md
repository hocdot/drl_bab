# Verifying Neural Networks with Reinforcement Learning

Formal verification can play a key role in ensuring the reliability of Deep Neural Networks (DNNs) deployed in safety-critical systems.
Modern DNN verifiers employ a branch-and-bound framework,
which alternates between branching (splitting into smaller subproblems) and
bounding (pruning subproblems) to efficiently explore the verification space.
However, existing branching heuristics make greedy decisions based on static scoring functions.
They do not anticipate long-term efficiency or leverage the growing availability of verification data to improve performance.

This work introduces RSB, a reinforcement learning framework that learns to refine baseline branching heuristics.
It trains an actor-critic architecture to maximize cumulative future rewards rather than immediate scores.
The actor generates attention weights from observations of raw neuron features and learned graph embeddings,
which rescale baseline heuristic scores to guide neuron branching.
Evaluation on 600 challenging instances demonstrates that RSB consistently outperforms state-of-the-art branching heuristics,
solving 11% more instances while reducing branch exploration by 50%.

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Reproduce results

- Remove the exiting `data/` foler

- Download generated [data.tar.xz](#)

- Extract `data.tar.xz` to create new `data/` folder

- Evaluate RSB/FSB/UPB/Polarity
```bash
python3 infer.py --config config/dnnv_fnn.yaml
python3 infer.py --config config/dnnv_cnn.yaml
python3 infer.py --config config/dnnv_distshift.yaml
```

- Plot results
```bash
python3 -m plot.create_table
python3 -m plot.plot_cactus_v2
python3 -m plot.plot_box
```

### 2. Generating FNN-based Instances & Pretrain GNN Embedding

- Generate random instances from existing ONNX at `data/dataset/dnnv/onnx`:
```bash
python3 -m prepare.generate_raw_instances --n_vnnlib_per_onnx 5000 --clean --config config/dnnv_fnn.yaml
```

- Filter out easy SAT instances:
```bash
python3 -m prepare.prepare_io --config config/dnnv_fnn.yaml
```

- Filter out easy UNSAT instances:
```bash
python3 -m prepare.prepare_hard_instances --config config/dnnv_fnn.yaml --use_early_stop
```

- Train GCN and extract embeddings:
```bash
python3 -m prepare.prepare_gnn_embedding --gcn_embed_batch_size 32 --config config/dnnv_fnn.yaml
```

- Split train and test sets:
```bash
python3 -m prepare.split_train_test --config config/dnnv_fnn.yaml
```

### 3. Train RSB

```bash
python3 train.py --mode train --config config/dnnv_fnn.yaml --use_early_stop
```

### 4. Evaluate RSB/FSB/UPB/Polarity

```bash
python3 infer.py --config config/dnnv_fnn.yaml
```

### 3. Generating CNN-based Instances & Evaluating RSB/FSB/UPB/Polarity

```bash
python3 -m prepare.generate_raw_instances --n_vnnlib_per_onnx 500 --clean --config config/dnnv_cnn.yaml
python3 -m prepare.prepare_io --config config/dnnv_cnn.yaml
python3 -m prepare.prepare_hard_instances --config config/dnnv_cnn.yaml --use_early_stop
python3 -m prepare.prepare_gnn_embedding --gcn_embed_batch_size 32 --config config/dnnv_cnn.yaml
cp data/dnnv_cnn_new/dnnv/instances_hard.csv data/dnnv_cnn_new/dnnv/instances_test.csv
python3 infer.py --config config/dnnv_cnn.yaml
```

### 4. Generating Sigmoid-based Instances & Evaluating RSB/FSB/UPB/Polarity

```bash
python3 -m prepare.generate_raw_distshift --n_vnnlib_per_onnx 500 --clean --config config/dnnv_distshift.yaml
python3 -m prepare.prepare_io --config config/dnnv_distshift.yaml
python3 -m prepare.prepare_hard_instances --config config/dnnv_distshift.yaml --use_early_stop
python3 -m prepare.prepare_gnn_embedding --gcn_embed_batch_size 32 --config config/dnnv_distshift.yaml
cp data/dnnv_distshift/dnnv/instances_hard.csv data/dnnv_distshift/dnnv/instances_test.csv
python3 infer.py --config config/dnnv_distshift.yaml
```

