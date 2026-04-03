import json
from pathlib import Path


def load_results():
    path = Path('experiments/identification/results/full_simulation.json')
    assert path.exists(), "Run population results first"
    return json.loads(path.read_text())


def test_type_b_zero_for_structural():
    js = load_results()
    # At k=3, structural methods should be exact
    row = js['B']['3']
    for m in ['Oracle', 'AIRL+anchors', 'IQ / GLADIUS']:
        assert abs(row[m]) < 1e-9


def test_paywall_ordering():
    js = load_results()
    row = js['F']
    assert row['Reduced-form'] > row['AIRL-no-anchors'] >= 0
    assert row['AIRL+anchors'] == 0 == row['Oracle'] == row['IQ / GLADIUS']


def test_metrics_present():
    js = load_results()
    assert 'metrics' in js and 'B' in js['metrics'] and 'F' in js['metrics']
    rf = js['metrics']['B']['3']['Reduced-form']
    # Check a few keys
    for k in ['uniform_l1', 'uniform_kl', 'regret_uniform']:
        assert k in rf

