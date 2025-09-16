from torch.utils.data import Dataset

import numpy as np
import tqdm
import time
import os

from arguments import get_args

def generate_hard_milp(args):
    """
    Generate medium-to-hard 0-1 ILP instances (MDKP-like).

    - A has shape (n_constraint, n_var)
    - bounds enforce binary variables (0 or 1)
    - profits are weakly correlated with weights to make LP relaxations less decisive
    - each capacity is 45%-60% of the total weight, creating tighter constraints
    """
    n_var = int(args.n_var)
    n_constraint = int(args.n_constraint)
    # Integer weights per constraint/variable (positive coefficients)
    A = np.random.randint(1, 101, size=(n_constraint, n_var))
    # Weakly correlated profits: sum of weights across constraints + small noise
    weight_sum_per_var = A.sum(axis=0)
    noise = np.random.randint(0, 21, size=n_var)
    c = weight_sum_per_var + noise
    # Medium-to-hard density per constraint (tighter but feasible)
    density = np.random.uniform(0.45, 0.60, size=n_constraint)
    b = (density * A.sum(axis=1)).astype(int)
    # Ensure strictly positive capacities
    b = np.maximum(b, 1)
    # Binary bounds
    bounds = np.array([(0, 1)] * n_var)
    # Return as numpy arrays
    return c.astype(np.float64), A.astype(np.float64), b.astype(np.float64), bounds


class MILPDataset(Dataset):

    def __init__(self, args):
        # save args
        self.args = args

    def prepare(self):
        # extract args
        args = self.args
        # check if data exists
        n_instance = eval(f'args.n_{self.args.mode}_episode')
        label = f'{args.dataset}_{self.args.mode}_{args.n_var}_{args.n_constraint}_{n_instance}.npz'
        path = os.path.join(args.dataset_dir, label)
        if os.path.exists(path):
            data   = np.load(path, allow_pickle=True)
            self.c      = np.array(data['c'])
            self.A      = np.array(data['A'])
            self.b      = np.array(data['b'])
            self.bounds = np.array(data['bounds'])
        else: # if not, generate new data
            # generate 2d coordinates for inputs
            print(f'[+] preparing {label}')
            tic = time.time()
            self.c, self.A, self.b, self.bounds = [], [], [], []
            for i in tqdm.tqdm(range(n_instance)):
                c, A, b, bounds = generate_hard_milp(args)
                self.c.append(c)
                self.A.append(A)
                self.b.append(b)
                self.bounds.append(bounds)
            self.c = np.array(self.c)
            self.A = np.array(self.A)
            self.b = np.array(self.b)
            self.bounds = np.array(self.bounds)
            # save data
            np.savez_compressed(path, c=self.c, A=self.A, b=self.b, bounds=self.bounds)
            toc = time.time()
            dt  = toc - tic
            print(f'    - saved {path=} {dt=:0.1f} (s)')

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        item = {'c': self.c[i], 'A': self.A[i], 'b': self.b[i], 'bounds': self.bounds[i]}
        return item
    
if __name__ == '__main__':
    args = get_args()
    dataset = MILPDataset(args)
    dataset.prepare()