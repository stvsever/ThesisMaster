#!/usr/bin/env python3
"""Build the portable, source-checked ontology snapshot. No model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
ONTO = REPO / 'src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw'
SOURCES = {
    'predictor': ONTO / 'PREDICTOR/steps/01_raw/aggregated/PREDICTOR_ontology.json',
    'criterion': ONTO / 'CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json',
    'person': ONTO / 'PERSON/PERSON.json',
    'context': ONTO / 'CONTEXT/CONTEXT.json',
    'hapa': ONTO / 'HAPA/HAPA.json',
}

def leaves(value, path=()):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    else:
        yield list(path)


def main():
    trees = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
    paths = {key: list(leaves(tree)) for key, tree in trees.items()}
    cases = json.loads((ROOT / 'data/cases.authoring.json').read_text())
    for case in cases:
        for criterion in case['criteria']:
            query = criterion.pop('match')
            matches = [p for p in paths['criterion'] if all(part in ' / '.join(p) for part in query)]
            if len(matches) != 1:
                raise ValueError(f"{case['id']} / {criterion['label']}: {len(matches)} criterion matches: {matches[:3]}")
            criterion['path'] = matches[0]
            assert criterion['span'] in case['complaint'], criterion['span']
        for candidate in case['candidates']:
            query = candidate.pop('match')
            matches = [p for p in paths['predictor'] if all(part in ' / '.join(p) for part in query)]
            if len(matches) != 1:
                raise ValueError(f"{case['id']} / {candidate['label']}: {len(matches)} predictor matches: {matches[:3]}")
            candidate['path'] = matches[0]
    snapshot = {
        'provenance': {key: {'source': str(path.relative_to(REPO)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'leaves': len(paths[key])} for key, path in SOURCES.items()},
        'predictor': trees['predictor'],
        'maxPredictorDepth': max(map(len, paths['predictor'])),
        'cases': cases,
        'notice': 'Authored fictional cases and simulated data. Ontology paths are repository-derived. This is an explanatory demonstration, not an engine run, diagnosis, or clinical efficacy study.'
    }
    (ROOT / 'data/snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f"Verified {len(cases)} cases, {sum(len(c['criteria']) for c in cases)} criteria, {sum(len(c['candidates']) for c in cases)} candidate paths; {len(paths['predictor'])} PREDICTOR leaves.")

if __name__ == '__main__':
    main()
