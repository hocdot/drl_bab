import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

SOLVER_DISPLAY = (
    # ('RandomDecider', 'Random'),
    ('PolarityDecider', r'\textsc{Polarity}'),
    ('UpbDecider', r'\textsc{Upb}'),
    ('FsbDecider', r'\textsc{Fsb}'),
    ('SACDecider', r'\textsc{Rsb}'),
)


BENCHMARK_TO_CONFIG = {
    'fnn': 'config/dnnv_fnn.yaml',
    'cnn': 'config/dnnv_cnn.yaml',
    'sigmoid': 'config/dnnv_distshift.yaml',
}

METRIC_COLS = ['value', 'done', 'decision_time', 'iteration', 'total_time', 'active_neuron']

EASY_THRESHOLD = -1000
TIMEOUT_THRESHOLD = -float('inf')
INSTANCES_PER_BENCHMARK = 50

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_EXPORT_DIR = os.path.join(CURRENT_DIR, 'csv')
os.makedirs(CSV_EXPORT_DIR, exist_ok=True)


FIGURE_DIR = os.path.join(CURRENT_DIR, 'figure')
FIGURE_DIR = '../../figure/evaluation/'
os.makedirs(FIGURE_DIR, exist_ok=True)


BENCHMARK_PANELS = (
    ('all', None),
    *[(k, k) for k in BENCHMARK_TO_CONFIG],
)

PANEL_TITLE = {
    'all': 'All',
    'fnn': 'FNN',
    'cnn': 'CNN',
    'sigmoid': 'Sigmoid',
}
