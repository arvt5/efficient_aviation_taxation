"""Shared export settings for the paper's Part I figures."""
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})


def save(fig, folder: Path, name: str):
    folder.mkdir(exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(folder / f'{name}.{extension}', dpi=160)
    plt.close(fig)
