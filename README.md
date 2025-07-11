# TODO

# Install neuralsat
```sh
# git submodule add -b heuristic git@github.com:dynaroars/neuralsat.git neuralsat
git submodule update --init --recursive
cd neuralsat
pip3 install -e .
```

## Hai Duong
- [ ] CNN problems ONNX + VNNLIB
- [ ] 1 network, N checkpoints (limited to N unstable neurons)
- [ ] Node features: [(l1, u1), ..., (ln, un)] + (lo, \_). If lo > 0 => branch is verified ---- lo <=> tour length
- [ ] Objective function: sum(lo) + n_problems

## Thanh Le
- [ ] Rebuild codebase
