# Neuralsat + Heuristic branch and bound

### Installation
```bash
pip3 install -r requirements.txt
```

### To run the main program
```bash
make main
```

### NOTE:
- [ ] Note that gen dataset for pretrain critic depend on action selection strategy; therefore, pretrain model may not be optimal in estimating long-term reward for other strategies
- [x] The buffer should be saved in RAM instead of VRAM
- [x] Add net_id into dataset

### TODO: truoc het supervised
- [x] Full reward feedback for smaller problem for critic (now not depend on action), train offline
- [x] GNN (i.e. GraphSAGE from torch_geometric) for inference reward (select from current node and neighboring node)
- [x] Using critic to choose action
- [x] Clean code of fc solver
- [x] Fix cai set done=False
- [x] Full pipeline on set of diverse topo + property
- [x] Ve ham loss, visualize cac cai khac trong test
- [x] Sau preprocess, check if exist unsat example, then if it is the case, quit. Write attacker from outside
- [x] From attacker.attacker import Attacker, sau dong 47
- [x] Do not load all model at start
- [x] Split train/test dataset
- [x] Worst bound less than -50, then solver should give up
- [x] Refactor solver to inherit from base solver
- [x] Measure and report runtime
- [x] Integrate Actor va Critic
- [x] Actor using GNN
- [x] GNN masked xin via Actor Mask + Mask
- [x] Making slide
- [x] Check lai greedy (urgent) (max-max -> max-min)
- [x] Gen data -1.5, 800 train, test 200, ra vnnlib text mỗi file một objective, attack luôn hộ, check luôn hộ 1.5
- [x] Gen 2,3,4,5,6 layers
- [x] Terminate state trick
- [-] Learn on heuristic, generate mixed between heuristic and greedy
- [ ] In step of greedy_2 and heuristic as lower and upper bound
- [ ] Choose top-k score, convert to iv action
- [ ] Normalize data offline
- [ ] Learn on sequence of previous states
- [ ] Solve data skewness problem, select problem with less n_visited more for balance distribution
- [ ] Importance sampling (lấy actor cũ để làm prob_old)
- [ ] Distributional shift reduction by KL-div
- [ ] Model-based RL
- [ ] Greedy biet truoc het next step
- [ ] Train model fancy
- [ ] Scale lên problem to

### BASELINE
- [ ] Heuristic bunel
- [ ] GNN, email hoi
- [ ] Random

### LAM SAU
- [ ] Then, RL on top-k for larger scale problem
- [ ] Gradually reduce greedy action instead of abruptly reduce for actor
- [ ] Them VNNLib vao input cua Critic
- [ ] Integrate top-k Greedy with long-term reward in scoring method

### Idea cho bai sau
- [ ] Verification for federated learning
- [ ] Verification for finding more robust Multiplier in CMDP
