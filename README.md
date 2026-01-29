# DRL-based Decision Heuristic for DNN Verification

Formal verification can play a key role in ensuring the reliability of Deep Neural Networks (DNN) deployed in safety-critical systems. Modern DNN verifiers employ branch-and-bound framework, which alternates between bounding (to prune subproblems) and branching (to split into smaller subproblems), to efficiently explore the verification space. However, existing branching heuristics make locally greedy decisions based on static scoring functions without anticipating long-term efficiency, and do not leverage the growing availability of verification data to improve performance.

This work introduces RSB, a reinforcement learning framework that learns to refine baseline branching heuristics. It trains a PointerNet-based actor-critic architecture to maximize cumulative future rewards rather than immediate scores. The actor generates attention weights from observations of raw neuron features and learned graph embeddings, which rescale baseline heuristic scores to guide neuron branching. Evaluation on 1,500 challenging instances demonstrates that RSB consistently outperforms state-of-the-art branching heuristics, solving 16% more instances while reducing branch exploration by 40%.


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
