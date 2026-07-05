import pandas as pd
import numpy as np

# Load your POI data
df_pois = pd.read_csv('points_of_interest.csv')

# ============================================================================
# CALCULATE UTILITY SCORES
# ============================================================================

# 1. Normalise Google Rating (0-10 scale)
df_pois['rating_norm'] = df_pois['google_rating'] / 5.0 * 10

# 2. Normalise Review Count (log scale to handle outliers)
df_pois['review_norm'] = np.log1p(df_pois['review_count'])
df_pois['review_norm'] = df_pois['review_norm'] / df_pois['review_norm'].max() * 10

# 3. Zone accessibility score
zone_weight = {'A': 10, 'B': 8, 'C': 6}
df_pois['zone_score'] = df_pois['zone'].map(zone_weight)

# 4. Indoor comfort bonus
df_pois['indoor_score'] = df_pois['indoor_ratio'] * 10

# 5. Cultural/historical bonus
cultural_categories = ['Cultural / Historical', 'Modern Landmarks', 'Religious Sites']
df_pois['cultural_bonus'] = df_pois['category'].apply(
    lambda x: 10 if x in cultural_categories else 5
)

# 6. Calculate final utility score (0-10 scale)
df_pois['utility_score'] = (
    0.30 * df_pois['rating_norm'] +      # 30% weight
    0.25 * df_pois['review_norm'] +      # 25% weight
    0.20 * df_pois['indoor_score'] +     # 20% weight
    0.15 * df_pois['zone_score'] / 10 +  # 15% weight
    0.10 * df_pois['cultural_bonus'] / 10 # 10% weight
)

# Normalise to ensure scores are 0-10
df_pois['utility_score'] = df_pois['utility_score'] / df_pois['utility_score'].max() * 10

# ============================================================================
# SAVE — Store together with POI data
# ============================================================================

df_pois.to_csv('poi_data_with_utility.csv', index=False)
print("Saved: poi_data_with_utility.csv")

# ============================================================================
# VERIFY RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("TOP 10 POIs BY UTILITY SCORE")
print("=" * 60)
print(df_pois[['poi_id', 'name', 'google_rating', 'review_count', 'utility_score']]
      .sort_values('utility_score', ascending=False)
      .head(10)
      .to_string(index=False))

print("\n" + "=" * 60)
print("BOTTOM 5 POIs BY UTILITY SCORE")
print("=" * 60)
print(df_pois[['poi_id', 'name', 'google_rating', 'review_count', 'utility_score']]
      .sort_values('utility_score', ascending=True)
      .head(5)
      .to_string(index=False))

# Summary statistics
print("\n" + "=" * 60)
print("UTILITY SCORE STATISTICS")
print("=" * 60)
print(f"Min: {df_pois['utility_score'].min():.2f}")
print(f"Max: {df_pois['utility_score'].max():.2f}")
print(f"Mean: {df_pois['utility_score'].mean():.2f}")
print(f"Median: {df_pois['utility_score'].median():.2f}")