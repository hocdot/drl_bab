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

input: 1
output: [0, 1]

input: 3
output: [0, 1, 2, 3, 5]

output 0, 1, 2 index of left subproblem of input 0, 1, 2
output 3, 5 index of right subproblem of input 0, 2

```python
for i in output:
    input[i] = i % len(input)
    if i >= len(input):
        right branch (-)
    else i < len(input):
        left branch (+)
# sort by desc
```
