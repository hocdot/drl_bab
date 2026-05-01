import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import FIGURE_DIR

_TRAIN_LOSS_TABLE = [
    (0, 3_000, 1.47e-03, 3.31e-01),
    (4_000, 96_000, 4.42e-04, 3.00e-02),
    (8_000, 189_000, 1.17e-04, 3.41e-03),
    (12_000, 282_000, 2.17e-04, 5.19e-03),
    (16_000, 377_000, 8.95e-05, 8.16e-04),
    (20_000, 471_000, 4.18e-05, 2.18e-04),
    (24_000, 565_000, 1.89e-05, 4.32e-05),
    (28_000, 659_000, 8.44e-06, 6.44e-06),
    (32_000, 752_000, 6.71e-06, 4.64e-06),
    (36_000, 844_000, 4.90e-06, 1.58e-06),
    (40_000, 937_000, 4.91e-06, 1.40e-06),
    (44_000, 1_032_000, 3.91e-06, 1.25e-06),
    (48_000, 1_127_000, 4.39e-06, 1.40e-06),
    (52_000, 1_220_000, 4.72e-06, 1.35e-06),
    (56_000, 1_313_000, 5.03e-06, 1.48e-06),
    (60_000, 1_405_000, 5.29e-06, 1.76e-06),
    (64_000, 1_499_000, 4.31e-06, 1.32e-06),
    (68_000, 1_592_000, 4.78e-06, 1.55e-06),
    (72_000, 1_684_000, 5.35e-06, 1.46e-06),
    (76_000, 1_778_000, 4.08e-06, 1.10e-06),
    (80_000, 1_873_000, 5.22e-06, 1.64e-06),
    (84_000, 1_965_000, 4.44e-06, 1.42e-06),
    (88_000, 2_058_000, 5.53e-06, 1.56e-06),
    (92_000, 2_152_000, 4.20e-06, 1.34e-06),
]


def _train_loss_dataframe() -> pd.DataFrame:
    rows = _TRAIN_LOSS_TABLE
    return pd.DataFrame(
        rows,
        columns=['episode', 'global_step', 'actor_loss', 'critic_loss'],
    )


def plot_train_loss(
    x: str = 'global_step',
    fmt: str = 'pdf',
) -> str:
    df = _train_loss_dataframe()
    long_ = df.melt(
        id_vars=[x],
        value_vars=['actor_loss', 'critic_loss'],
        var_name='loss',
        value_name='value',
    )
    long_['loss'] = long_['loss'].str.replace('_loss', '', regex=False).str.title()
    xlabel = 'Global step' if x == 'global_step' else 'Episode'
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f'train_loss.{fmt}')

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=long_, x=x, y='value', hue='loss', markersize=3, dashes=False)
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(title='', frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close()
    print(f'[+] saved {path}')
    return path


if __name__ == '__main__':
    plot_train_loss(x='episode', fmt='pdf')
