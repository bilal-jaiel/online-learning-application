"""
Plots
------
Standardised plotting functions for experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A',
    '#F4A261', '#264653', '#A8DADC', '#6D6875',
]

mpl.rcParams.update({
    'font.family': 'monospace',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 130,
})


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')


def plot_cumulative_rewards(results_dict, title='Cumulative Reward',
                             filename='cumulative_reward.png', phase_boundaries=None):
    """
    results_dict : {algo_name: np.array of per-round rewards}
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (name, rewards) in enumerate(results_dict.items()):
        cum = np.cumsum(rewards)
        ax.plot(cum, label=name, color=PALETTE[idx % len(PALETTE)], linewidth=1.8)

    if phase_boundaries:
        for pb in phase_boundaries:
            ax.axvline(pb, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    _ax_style(ax, title, 'Round t', 'Cumulative Reward')
    ax.legend(fontsize=8, framealpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cumulative_regret(results_dict, oracle_reward_per_round,
                            title='Cumulative Regret',
                            filename='cumulative_regret.png',
                            phase_boundaries=None):
    """
    oracle_reward_per_round : scalar (expected per-round reward of oracle)
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (name, rewards) in enumerate(results_dict.items()):
        T = len(rewards)
        regret = np.cumsum(oracle_reward_per_round - np.array(rewards))
        ax.plot(regret, label=name, color=PALETTE[idx % len(PALETTE)], linewidth=1.8)

    if phase_boundaries:
        for pb in phase_boundaries:
            ax.axvline(pb, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    _ax_style(ax, title, 'Round t', 'Cumulative Regret')
    ax.legend(fontsize=8, framealpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_budget_consumption(costs_dict, total_budget,
                             title='Budget Consumption',
                             filename='budget_consumption.png',
                             phase_boundaries=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (name, costs) in enumerate(costs_dict.items()):
        frac = np.cumsum(costs) / total_budget
        ax.plot(frac, label=name, color=PALETTE[idx % len(PALETTE)], linewidth=1.8)

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Budget limit')

    if phase_boundaries:
        for pb in phase_boundaries:
            ax.axvline(pb, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    _ax_style(ax, title, 'Round t', 'Fraction of Budget Used')
    ax.legend(fontsize=8, framealpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_comparison_table(summary_dict, filename='comparison_table.png'):
    """
    summary_dict : {algo_name: {metric: value}}
    """
    algo_names = list(summary_dict.keys())
    metrics = ['total_reward', 'total_cost', 'win_rate', 'budget_used_frac', 'avg_reward']
    labels  = ['Total Reward', 'Total Cost', 'Win Rate', 'Budget Used', 'Avg Reward/Round']

    data = np.zeros((len(algo_names), len(metrics)))
    for i, name in enumerate(algo_names):
        for j, m in enumerate(metrics):
            data[i, j] = round(summary_dict[name].get(m, 0), 4)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.5 * len(algo_names)))
    ax.axis('off')
    tbl = ax.table(
        cellText=data,
        rowLabels=algo_names,
        colLabels=labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.4)
    ax.set_title('Algorithm Comparison', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
