"""Offline RDAP counterfactual audit; never fits or writes model/dataset files."""
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    before = pd.read_csv(ROOT / 'results/google_before_rdap_fix.csv')
    after = pd.read_csv(ROOT / 'results/google_after_rdap_fix.csv')
    assert len(before) == len(after) == 1
    differences = [
        {'feature': name, 'before': str(before[name].iloc[0]),
         'after': str(after[name].iloc[0])}
        for name in before if not before[name].equals(after[name])
    ]
    rows = []
    hashes = {}
    for label, model_name, prep_name in (
        ('original80', 'lightgbm_phishing_model', 'training_preprocessor'),
        ('tuned80', 'lightgbm_80_tuned_model', 'training_preprocessor_80_tuned'),
        ('tuned40', 'lightgbm_40_tuned_model', 'training_preprocessor_40_tuned'),
    ):
        paths = [ROOT / 'models' / (name + '.joblib')
                 for name in (model_name, prep_name)]
        for path in paths:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        model, prep = (joblib.load(path) for path in paths)
        positive_index = list(model.classes_).index(1)
        for baseline, frame in (('before', before), ('after', after)):
            for value in (0, 1):
                changed = frame.copy(deep=True)
                changed['rdap_abuse_count'] = value
                other = [c for c in frame if c != 'rdap_abuse_count']
                pd.testing.assert_frame_equal(frame[other], changed[other])
                transformed = prep.transform(changed[list(prep.feature_names_in_)])
                probability = float(model.predict_proba(transformed)[0, positive_index])
                rows.append({'model': label, 'baseline': baseline,
                             'rdap_abuse_count': value,
                             'phishing_probability': probability})
        for path in paths:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
    train = pd.read_csv(ROOT / 'processed/train_balanced_raw.csv',
                        usecols=['label', 'rdap_abuse_count', 'rdap_has_abuse_contact'])
    distribution = train.groupby(
        ['label', 'rdap_abuse_count', 'rdap_has_abuse_contact'], dropna=False
    ).size().reset_index(name='rows')
    report = {'export_differences': differences, 'controlled_predictions': rows,
              'training_distribution': distribution.to_dict(orient='records'),
              'model_and_preprocessor_sha256': hashes,
              'scope': 'Only abuse count changes in each counterfactual. Contact flag stays fixed, intentionally isolating count. No network, fitting, calibration, or dataset mutation.'}
    # Exclusive creation protects any earlier audit output.
    output = ROOT / 'results/rdap_controlled_audit.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Saved:', output)


if __name__ == '__main__':
    main()
