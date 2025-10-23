import pandas as pd
import random
import os

from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    
    all_csv_path = os.path.join(args.dataset_dir, args.dataset, 'instances_hard.csv')
    all_lines = pd.read_csv(all_csv_path, header=None).values.tolist()
    random.shuffle(all_lines)
    train_lines = all_lines[:int(len(all_lines) * 0.8)]
    test_lines = all_lines[int(len(all_lines) * 0.8):]
    
    train_path = os.path.join(args.dataset_dir, args.dataset, 'instances_train.csv')
    test_path = os.path.join(args.dataset_dir, args.dataset, 'instances_test.csv')
    with open(train_path, 'w') as f:
        for line in train_lines:
            print(','.join(line), file=f)
    with open(test_path, 'w') as f:
        for line in test_lines:
            print(','.join(line), file=f)
    print(f'{len(train_lines)=} {len(test_lines)=}')