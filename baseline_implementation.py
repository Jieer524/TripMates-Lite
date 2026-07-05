"""
BASELINE IMPLEMENTATIONS FOR ADAPTIVE URBAN TOURISM PLANNING
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TOTAL_TIME = 480  # 8 hours (480 minutes)
START_POI_ID = 1  # Start from Dataran Merdeka
DISRUPTION_TIME = 120  # Disruption occurs at 120 minutes

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load POI data
df_pois = pd.read_csv('poi_data_with_utility.csv')
print(f"Loaded {len(df_pois)} POIs")

# Load travel matrix - index_col=0 is CRITICAL
df_time = pd.read_csv('travel_matrix_walk.csv', index_col=0)
print(f"Loaded {df_time.shape[0]}x{df_time.shape[1]} travel matrix")

# Clean column and index names
df_time.columns = [col.strip() for col in df_time.columns]
df_time.index = [idx.strip() for idx in df_time.index]

# Ensure all values are numeric
df_time = df_time.apply(pd.to_numeric, errors='coerce')

# Display verification
print("\nTravel matrix shape:", df_time.shape)
print("Sample travel times from Dataran Merdeka (ID: 1):")
print(df_time.loc['1: Dataran Merdeka'].head(10))

# Get mandatory POIs
mandatory_pois = df_pois[df_pois['mandatory'].astype(str).str.upper() == 'YES']['poi_id'].tolist()
print(f"\nMandatory POIs: {mandatory_pois}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_poi_name(poi_id):
    """Get POI name by ID"""
    try:
        return df_pois[df_pois['poi_id'] == poi_id]['name'].values[0]
    except:
        return f"POI_{poi_id}"

def get_poi_utility(poi_id):
    """Get utility score by ID"""
    try:
        val = df_pois[df_pois['poi_id'] == poi_id]['utility_score'].values[0]
        return float(val) if not pd.isna(val) else 0.0
    except:
        return 0.0

def get_visit_duration(poi_id):
    """Get visit duration by ID"""
    try:
        val = df_pois[df_pois['poi_id'] == poi_id]['visit_duration_min'].values[0]
        # Handle string values like "90 min"
        if isinstance(val, str):
            val = val.replace(' min', '').strip()
        return float(val) if not pd.isna(val) else 60.0
    except:
        return 60.0

def get_travel_time(from_id, to_id):
    """Get walking travel time between two POIs"""
    if from_id == to_id:
        return 0.0
    
    try:
        from_name = f"{from_id}: {get_poi_name(from_id)}"
        to_name = f"{to_id}: {get_poi_name(to_id)}"
        
        # Try exact match
        if from_name in df_time.index and to_name in df_time.columns:
            val = df_time.loc[from_name, to_name]
            return float(val) if not pd.isna(val) else 9999.0
        
        # Try partial match
        for idx in df_time.index:
            if str(from_id) in idx:
                for col in df_time.columns:
                    if str(to_id) in col:
                        val = df_time.loc[idx, col]
                        return float(val) if not pd.isna(val) else 9999.0
        
        return 9999.0
        
    except Exception as e:
        print(f"  Warning: Travel time error ({from_id}->{to_id}): {e}")
        return 9999.0

def calculate_metrics(itinerary, total_time_used, total_pois=50, total_time=480):
    """Calculate evaluation metrics"""
    
    if not itinerary:
        return {
            'ICR': 0, 'AUA': 0, 'TTT': 0, 'OEM': 0, 'RL': 0,
            'POI_Count': 0, 'Total_Utility': 0, 'Time_Used': 0
        }
    
    icr = (len(itinerary) / total_pois) * 100
    
    total_utility = sum(get_poi_utility(pid) for pid in itinerary)
    max_utility = df_pois['utility_score'].sum()
    aua = (total_utility / max_utility) * 100 if max_utility > 0 else 0
    
    total_visit_time = sum(get_visit_duration(pid) for pid in itinerary)
    ttt = total_time_used - total_visit_time
    
    return {
        'ICR': round(icr, 2),
        'AUA': round(aua, 2),
        'TTT': round(ttt, 1),
        'OEM': 0,
        'RL': 0,
        'POI_Count': len(itinerary),
        'Total_Utility': round(total_utility, 2),
        'Time_Used': round(total_time_used, 1)
    }

def print_itinerary(itinerary, total_time_used, title="ITINERARY"):
    """Pretty print an itinerary"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Total time used: {total_time_used:.1f} / {TOTAL_TIME} minutes")
    print(f"Number of POIs: {len(itinerary)}")
    print("\nOrder of visit:")
    
    cumulative_time = 0
    prev_id = START_POI_ID
    
    for i, poi_id in enumerate(itinerary, 1):
        name = get_poi_name(poi_id)
        utility = get_poi_utility(poi_id)
        duration = get_visit_duration(poi_id)
        travel = get_travel_time(prev_id, poi_id) if i > 1 else 0
        cumulative_time += travel + duration
        
        mandatory = "YES" if poi_id in mandatory_pois else "NO"
        
        print(f"  {i:2d}. {name}")
        print(f"      (ID: {poi_id}, Utility: {utility:.2f}, Duration: {duration}min, Mandatory: {mandatory})")
        if i > 1:
            print(f"      Travel from previous: {travel:.1f}min")
        
        prev_id = poi_id
    
    print(f"\nCumulative time at end: {cumulative_time:.1f} / {TOTAL_TIME} minutes")

# ============================================================================
# BASELINE 1: STATIC UTILITY
# ============================================================================

def baseline_static_utility():
    """Greedy algorithm: Select POIs with highest utility score"""
    
    print("\n" + "=" * 70)
    print("BASELINE 1: STATIC UTILITY")
    print("=" * 70)
    print("Greedy selection by highest utility score")
    
    start_time = time.time()
    
    itinerary = []
    current_time = 0.0
    current_location = START_POI_ID
    visited = set()
    
    # Step 1: Add mandatory POIs first (in order of utility)
    mandatory_sorted = sorted(mandatory_pois, 
                             key=lambda x: get_poi_utility(x), 
                             reverse=True)
    
    for poi_id in mandatory_sorted:
        travel_time = get_travel_time(current_location, poi_id)
        visit_duration = get_visit_duration(poi_id)
        
        if current_time + travel_time + visit_duration <= TOTAL_TIME:
            itinerary.append(poi_id)
            visited.add(poi_id)
            current_time += travel_time + visit_duration
            current_location = poi_id
        else:
            print(f"  Warning: Could not include mandatory POI {poi_id} ({get_poi_name(poi_id)})")
            print(f"    Needed: {travel_time + visit_duration:.1f}min, Remaining: {TOTAL_TIME - current_time:.1f}min")
    
    # Step 2: Add optional POIs greedily by utility
    optional_pois = df_pois[~df_pois['poi_id'].isin(visited)]['poi_id'].tolist()
    optional_sorted = sorted(optional_pois, 
                            key=lambda x: get_poi_utility(x), 
                            reverse=True)
    
    for poi_id in optional_sorted:
        travel_time = get_travel_time(current_location, poi_id)
        visit_duration = get_visit_duration(poi_id)
        
        if current_time + travel_time + visit_duration <= TOTAL_TIME:
            itinerary.append(poi_id)
            visited.add(poi_id)
            current_time += travel_time + visit_duration
            current_location = poi_id
    
    elapsed_time = time.time() - start_time
    
    print_itinerary(itinerary, current_time, "STATIC UTILITY ITINERARY")
    
    metrics = calculate_metrics(itinerary, current_time)
    metrics['Algorithm'] = 'Static Utility'
    metrics['Runtime'] = round(elapsed_time, 3)
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key not in ['Algorithm', 'Runtime']:
            print(f"  {key}: {value}")
    print(f"  Runtime: {elapsed_time:.3f}s")
    
    return itinerary, current_time, metrics

# ============================================================================
# BASELINE 2: STATIC DISTANCE
# ============================================================================

def baseline_static_distance():
    """Greedy algorithm: Always visit the nearest unvisited POI"""
    
    print("\n" + "=" * 70)
    print("BASELINE 2: STATIC DISTANCE")
    print("=" * 70)
    print("Greedy selection by nearest neighbour")
    
    start_time = time.time()
    
    itinerary = []
    current_time = 0.0
    current_location = START_POI_ID
    visited = set()
    
    # Step 1: Add mandatory POIs first (in order of nearest distance)
    mandatory_sorted = sorted(mandatory_pois,
                             key=lambda x: get_travel_time(current_location, x))
    
    for poi_id in mandatory_sorted:
        travel_time = get_travel_time(current_location, poi_id)
        visit_duration = get_visit_duration(poi_id)
        
        if current_time + travel_time + visit_duration <= TOTAL_TIME:
            itinerary.append(poi_id)
            visited.add(poi_id)
            current_time += travel_time + visit_duration
            current_location = poi_id
    
    # Step 2: Add optional POIs greedily by nearest distance
    while True:
        optional_pois = df_pois[~df_pois['poi_id'].isin(visited)]['poi_id'].tolist()
        if not optional_pois:
            break
        
        nearest_poi = None
        nearest_time = float('inf')
        
        for poi_id in optional_pois:
            travel_time = get_travel_time(current_location, poi_id)
            if travel_time < nearest_time:
                nearest_time = travel_time
                nearest_poi = poi_id
        
        if nearest_poi is None:
            break
        
        visit_duration = get_visit_duration(nearest_poi)
        
        if current_time + nearest_time + visit_duration <= TOTAL_TIME:
            itinerary.append(nearest_poi)
            visited.add(nearest_poi)
            current_time += nearest_time + visit_duration
            current_location = nearest_poi
        else:
            # If nearest doesn't fit, try next nearest
            remaining_pois = [p for p in optional_pois if p != nearest_poi]
            found = False
            for poi_id in remaining_pois:
                travel_time = get_travel_time(current_location, poi_id)
                visit_duration = get_visit_duration(poi_id)
                if current_time + travel_time + visit_duration <= TOTAL_TIME:
                    itinerary.append(poi_id)
                    visited.add(poi_id)
                    current_time += travel_time + visit_duration
                    current_location = poi_id
                    found = True
                    break
            if not found:
                break
    
    elapsed_time = time.time() - start_time
    
    print_itinerary(itinerary, current_time, "STATIC DISTANCE ITINERARY")
    
    metrics = calculate_metrics(itinerary, current_time)
    metrics['Algorithm'] = 'Static Distance'
    metrics['Runtime'] = round(elapsed_time, 3)
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key not in ['Algorithm', 'Runtime']:
            print(f"  {key}: {value}")
    print(f"  Runtime: {elapsed_time:.3f}s")
    
    return itinerary, current_time, metrics

# ============================================================================
# BASELINE 3: ADAPTIVE (NO THERMAL)
# ============================================================================

def baseline_adaptive_no_thermal():
    """Static Utility + dynamic replanning (no thermal cost)"""
    
    print("\n" + "=" * 70)
    print("BASELINE 3: ADAPTIVE (NO THERMAL)")
    print("=" * 70)
    print("Static Utility with dynamic replanning (no thermal cost)")
    print(f"Disruption at {DISRUPTION_TIME} minutes")
    
    start_time = time.time()
    
    # First, build a complete static utility itinerary
    static_itinerary, static_time, _ = baseline_static_utility()
    
    # Simulate progress until disruption
    current_time = 0.0
    current_location = START_POI_ID
    visited = set()
    itinerary = []
    
    # Visit POIs until disruption time
    for poi_id in static_itinerary:
        if poi_id not in visited:
            travel_time = get_travel_time(current_location, poi_id)
            visit_duration = get_visit_duration(poi_id)
            
            if current_time + travel_time + visit_duration <= DISRUPTION_TIME:
                itinerary.append(poi_id)
                visited.add(poi_id)
                current_time += travel_time + visit_duration
                current_location = poi_id
            else:
                current_time += travel_time
                print(f"\nDisruption occurred at {current_time:.1f} minutes")
                print(f"Current location: {get_poi_name(current_location)}")
                break
    
    # Phase 2: Replan from current location (greedy by utility)
    print(f"\nReplanning from {get_poi_name(current_location)}...")
    replan_start = time.time()
    
    remaining_pois = df_pois[~df_pois['poi_id'].isin(visited)]['poi_id'].tolist()
    remaining_pois_sorted = sorted(remaining_pois,
                                   key=lambda x: get_poi_utility(x),
                                   reverse=True)
    
    for poi_id in remaining_pois_sorted:
        travel_time = get_travel_time(current_location, poi_id)
        visit_duration = get_visit_duration(poi_id)
        
        if current_time + travel_time + visit_duration <= TOTAL_TIME:
            itinerary.append(poi_id)
            visited.add(poi_id)
            current_time += travel_time + visit_duration
            current_location = poi_id
    
    replan_time = time.time() - replan_start
    elapsed_time = time.time() - start_time
    
    print_itinerary(itinerary, current_time, "ADAPTIVE (NO THERMAL) ITINERARY")
    
    metrics = calculate_metrics(itinerary, current_time)
    metrics['Algorithm'] = 'Adaptive (No Thermal)'
    metrics['Runtime'] = round(elapsed_time, 3)
    metrics['Replan_Latency'] = round(replan_time, 3)
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key not in ['Algorithm', 'Runtime']:
            print(f"  {key}: {value}")
    print(f"  Runtime: {elapsed_time:.3f}s")
    print(f"  Replan Latency: {replan_time:.3f}s")
    
    return itinerary, current_time, metrics

# ============================================================================
# RUN ALL BASELINES
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING ALL BASELINES")
print("=" * 70)

results = []

# Baseline 1: Static Utility
try:
    itin1, time1, metrics1 = baseline_static_utility()
    results.append(metrics1)
except Exception as e:
    print(f"Error in Static Utility: {e}")
    import traceback
    traceback.print_exc()

# Baseline 2: Static Distance
try:
    itin2, time2, metrics2 = baseline_static_distance()
    results.append(metrics2)
except Exception as e:
    print(f"Error in Static Distance: {e}")
    import traceback
    traceback.print_exc()

# Baseline 3: Adaptive (No Thermal)
try:
    itin3, time3, metrics3 = baseline_adaptive_no_thermal()
    results.append(metrics3)
except Exception as e:
    print(f"Error in Adaptive (No Thermal): {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON SUMMARY")
print("=" * 70)

if results:
    df_results = pd.DataFrame(results)
    cols = ['Algorithm', 'POI_Count', 'Time_Used', 'ICR', 'AUA', 'TTT', 'Total_Utility', 'Runtime']
    if 'Replan_Latency' in df_results.columns:
        cols.append('Replan_Latency')
    df_results = df_results[[c for c in cols if c in df_results.columns]]
    
    print(df_results.to_string(index=False))
    df_results.to_csv('baseline_results.csv', index=False)
    print(f"\nResults saved to: baseline_results.csv")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if len(df_results) >= 2:
        best_icr = df_results.loc[df_results['ICR'].idxmax()]
        best_aua = df_results.loc[df_results['AUA'].idxmax()]
        fastest = df_results.loc[df_results['Runtime'].idxmin()]
        
        print(f"Highest ICR: {best_icr['Algorithm']} ({best_icr['ICR']:.2f}%)")
        print(f"Highest AUA: {best_aua['Algorithm']} ({best_aua['AUA']:.2f}%)")
        print(f"Fastest runtime: {fastest['Algorithm']} ({fastest['Runtime']:.3f}s)")
else:
    print("No results generated. Please check your data files.")

print("\n" + "=" * 70)
print("IMPLEMENTATION COMPLETE")
print("=" * 70)