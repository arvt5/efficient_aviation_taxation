"""Render the paper's archived occupancy-weighted price paths."""

import csv
import json
from pathlib import Path
import shutil
import subprocess

from .model import FractionalParams
from .price_paths import render_svg


def main():
    root = Path(__file__).resolve().parents[2]
    data = root / 'data/archived'
    output = root / 'outputs'
    output.mkdir(exist_ok=True)
    metadata = json.loads((data / 'price_path_metadata.json').read_text())
    converter = shutil.which('rsvg-convert')
    for name, meta in metadata.items():
        with (data / f'price_path_{name}.csv').open(newline='') as handle:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        scale = meta['usd_per_model_price_unit']
        svg = render_svg(rows, FractionalParams(**meta['params']), scale,
                         meta['ymin_usd'] / scale, meta['ymax_usd'] / scale)
        path = output / f'price_path_{name}.svg'
        path.write_text(svg)
        if converter:
            for suffix in ('pdf', 'png'):
                subprocess.run([converter, '-f', suffix, '-o', str(path.with_suffix('.' + suffix)), str(path)], check=True)
        print(path)
    if not converter:
        print('SVGs created. Install librsvg/rsvg-convert for PDF and PNG exports.')


if __name__ == '__main__':
    main()
