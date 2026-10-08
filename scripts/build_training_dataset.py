"""Build a scalable labelled UrbanPulse dataset from exported feature tables and labels.

Each feature file should contain cell_uid, row, col, year and numeric features.
Each label file should contain cell_uid and label (or geometry + label).
"""
import argparse
from pathlib import Path
import pandas as pd
from urbanpulse.config import CLASSES

ap=argparse.ArgumentParser()
ap.add_argument('--features-dir',required=True)
ap.add_argument('--labels-dir',required=True)
ap.add_argument('--out',default='data/training/urbanpulse_training.parquet')
args=ap.parse_args()

features=[]
for p in sorted(Path(args.features_dir).glob('*')):
    if p.suffix.lower() in {'.csv','.parquet','.pq'}:
        d=pd.read_parquet(p) if p.suffix.lower() in {'.parquet','.pq'} else pd.read_csv(p)
        features.append(d)
if not features: raise SystemExit('No feature tables found.')
f=pd.concat(features,ignore_index=True)

labels=[]
for p in sorted(Path(args.labels_dir).glob('*')):
    if p.suffix.lower() not in {'.csv','.geojson','.json'}: continue
    if p.suffix.lower()=='.csv': d=pd.read_csv(p)
    else:
        import geopandas as gpd
        d=gpd.read_file(p)
    if 'class' in d.columns and 'label' not in d.columns: d=d.rename(columns={'class':'label'})
    if 'label' not in d.columns: continue
    if 'cell_uid' not in d.columns: continue
    labels.append(d[['cell_uid','label']].dropna())
if not labels: raise SystemExit('No labels with cell_uid found.')
l=pd.concat(labels,ignore_index=True).drop_duplicates('cell_uid')
bad=sorted(set(l.label.astype(str))-set(CLASSES))
if bad: raise SystemExit(f'Unknown labels: {bad}')

out=f.merge(l,on='cell_uid',how='inner')
if out.empty: raise SystemExit('No labelled feature rows matched.')
out=out.drop_duplicates('cell_uid')
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
out.to_parquet(args.out,index=False)
print(f'Wrote {len(out)} labelled cells to {args.out}')
print(out.label.value_counts().to_string())
