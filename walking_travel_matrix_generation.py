"""
Generate Walking Travel Matrix — Fixed Version
Uses nx.shortest_path_length() to avoid edge unpacking errors
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import time

# ============================================================================
# LOAD POI DATA
# ============================================================================

df_pois = pd.read_csv('points_of_interest.csv')
df_pois['Latitude'] = pd.to_numeric(df_pois['Latitude'], errors='coerce')
df_pois['Longitude'] = pd.to_numeric(df_pois['Longitude'], errors='coerce')
df_pois = df_pois.dropna(subset=['Latitude', 'Longitude'])

print(f"Loaded {len(df_pois)} POIs")

# ============================================================================
# DOWNLOAD GRAPH
# ============================================================================

print("Downloading walking network...")
G = ox.graph_from_place('Kuala Lumpur, Malaysia', network_type='walk')
print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

# ============================================================================
# FIND NEAREST NODES
# ============================================================================

print("Finding nearest nodes...")
orig_nodes = []
for idx, row in df_pois.iterrows():
    node = ox.distance.nearest_nodes(G, row['Longitude'], row['Latitude'])
    orig_nodes.append(node)
    print(f"  {row['poi_id']}: {row['name']} -> Node {node}")

# ============================================================================
# CALCULATE DISTANCES USING nx.shortest_path_length()
# ============================================================================

print("\nCalculating pairwise distances...")
n = len(orig_nodes)
time_matrix = np.zeros((n, n))

# Walking speed: 1.4 m/s
WALKING_SPEED = 1.4  # m/s

start_time = time.time()

for i in range(n):
    for j in range(n):
        if i == j:
            time_matrix[i, j] = 0
        else:
            try:
                # Use nx.shortest_path_length() — handles edge weights automatically
                length_meters = nx.shortest_path_length(G, orig_nodes[i], orig_nodes[j], weight='length')
                time_matrix[i, j] = length_meters / (WALKING_SPEED * 60)
            except nx.NetworkXNoPath:
                time_matrix[i, j] = float('inf')
            except Exception as e:
                print(f"  Error ({i}->{j}): {e}")
                time_matrix[i, j] = float('inf')
    
    if (i + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Processed {i+1}/{n} POIs ({elapsed:.1f}s elapsed)")

print(f"Completed in {time.time() - start_time:.1f} seconds")

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

column_names = [f"{pid}: {name}" for pid, name in zip(df_pois['poi_id'], df_pois['name'])]
df_time = pd.DataFrame(time_matrix, index=column_names, columns=column_names)

# ============================================================================
# SAVE AND SUMMARIZE
# ============================================================================

df_time.to_csv('travel_matrix_walk.csv')
print(f"\nSaved: travel_matrix_walk.csv ({df_time.shape[0]}x{df_time.shape[1]})")

# Summary
times_flat = time_matrix[time_matrix < float('inf')].flatten()
print(f"\nTotal POI pairs: {n*n}")
print(f"Reachable pairs: {len(times_flat)}")
print(f"Unreachable pairs: {n*n - len(times_flat)}")

if len(times_flat) > 0:
    print(f"\nWalking time statistics (minutes):")
    print(f"  Min: {np.min(times_flat):.1f} min")
    print(f"  Max: {np.max(times_flat):.1f} min")
    print(f"  Mean: {np.mean(times_flat):.1f} min")
    print(f"  Median: {np.median(times_flat):.1f} min")

print("\nSample (first 10 rows, first 10 columns):")
print(df_time.iloc[:10, :10].round(1))