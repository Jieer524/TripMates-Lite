import pandas as pd
import numpy as np

# Load your data
df_pois = pd.read_csv('points_of_interest.csv')
df_time = pd.read_csv('travel_matrix_walk.csv', index_col=0)

print("=" * 60)
print("FINAL DATASET VALIDATION")
print("=" * 60)

# 1. POI count
print(f"\nPOIs in dataset: {len(df_pois)}")
print(f"POIs in matrix: {df_time.shape[0]}")

# 2. Check alignment
poi_ids = [str(x) for x in df_pois['poi_id'].values]
matrix_ids = [x.split(':')[0] for x in df_time.index.values]

print(f"\nPOI IDs in dataset: {poi_ids[:5]}...")
print(f"POI IDs in matrix: {matrix_ids[:5]}...")

# 3. Check if all POIs are in the matrix
all_present = all([pid in matrix_ids for pid in poi_ids])
print(f"\nAll POIs in matrix: {all_present}")

# 4. Mandatory POIs
mandatory = df_pois[df_pois['mandatory'] == 'YES']
print(f"\nMandatory POIs ({len(mandatory)}):")
for _, row in mandatory.iterrows():
    print(f"  {row['poi_id']}: {row['name']} ({row['zone']})")

# 5. Zone distribution
print(f"\nZone distribution:")
print(df_pois['zone'].value_counts().sort_index())

# 6. Category distribution
print(f"\nCategory distribution:")
print(df_pois['category'].value_counts())

print("\n" + "=" * 60)
print("DATASET READY FOR EXPERIMENTS")
print("=" * 60)