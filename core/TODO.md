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

## Thanh Le
- [x] Extract subgraph based on vnnlib
- [x] Cache list potential vnnlib
- [ ] Check if subgraph pruning is correct with CNN onnx
- [ ] Finally cache subgraph (sub edge index, edge weight)
- [ ] Idea: non-trainable embedding if it is faster
