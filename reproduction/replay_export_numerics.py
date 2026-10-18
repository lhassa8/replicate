"""Recalculate published merge/probability figures from preserved outputs only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from organism import audit_export_responses as original


def replay():
    base = ROOT / 'evidence/publication-support/export'
    saved = original.load(base / 'independent-response-audit.json')
    source = Path(original.__file__).read_bytes()
    original.require({'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()} == saved['source'],
                     'Numerical auditor differs from the historical audit source')
    report = original.load(base / 'export/report.json')
    merged = original.load(base / 'merge/report.json')
    original.require(report['prompts'] == merged['prompts'] and len(report['prompts']) == 3,
                     'Merge and export prompt identities differ')
    arrays = {name: np.load(base / 'merge' / (name + '-reference-logits.npy'), allow_pickle=False)
              for name in ('base', 'adapter', 'merged')}
    merge_error = float(np.abs(arrays['adapter'].astype(np.float64) - arrays['merged'].astype(np.float64)).max())
    original.require(abs(merge_error - saved['merge_error']) <= 1e-12, 'Merge figure differs')
    arms = {}
    for label in ('stock-f32', 'trained-f32', 'stock-q4', 'trained-q4'):
        directory = base / 'export' / label
        serving = original.load(directory / 'report.json')
        original.require(serving['failure'] is None and len(serving['requests']) == 3,
                         'Historical serving result differs')
        rows = np.stack([original.reconstruct_response(directory, item, prompt, arrays['base'].shape[1])
                         for item, prompt in zip(serving['requests'], report['prompts'])])
        actual = original.compare(arrays['base' if label.startswith('stock') else 'merged'], rows)
        for key, value in actual.items():
            original.require(np.allclose(value, saved['arms'][label][key], atol=1e-10, rtol=0),
                             'Published probability figure differs: ' + label + '/' + key)
        arms[label] = actual
    return {'scope': 'Published merge/probability numbers reconstructed from 12 saved responses; not full export admission or tensor-layout reconstruction',
            'merge_error': merge_error, 'arms': arms, 'saved_numerical_figures_match': True,
            'new_inference': False, 'training_rerun': False}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
