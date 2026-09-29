import json
import numpy as np
import os

print("Generating dashboard_aggregates.json...")

# 1. Event Intensity Evolution (from persistent_events.json)
try:
    print("Loading persistent_events.json (this may take a few seconds)...")
    with open('../final_model/event/persistent_events.json', 'r') as f:
        persistent = json.load(f)
    
    # We will average the intensity and probability across all events for each lead hour
    lead_stats = {}
    for ev in persistent:
        for obs in ev['observations']:
            lh = obs['lead_hour']
            if lh not in lead_stats:
                lead_stats[lh] = {'intensity_z': [], 'mean_probability': []}
            lead_stats[lh]['intensity_z'].append(obs['intensity_z'])
            lead_stats[lh]['mean_probability'].append(obs['mean_probability'])
    
    evolution = []
    for lh in sorted(lead_stats.keys()):
        evolution.append({
            'lead': f"{lh}h",
            'intensity': round(np.mean(lead_stats[lh]['intensity_z']), 2), # true z-score
            'probability': round(np.mean(lead_stats[lh]['mean_probability']) * 100, 1)
        })
    print("Evolution computed.")
except Exception as e:
    print("Error with persistent_events:", e)
    evolution = []

# 2. Monthly Distribution (from test_initialization_times.npy and test_event_probabilities.npy)
try:
    print("Loading npy files...")
    times = np.load('../final_model/event/test_initialization_times.npy', allow_pickle=True)
    probs = np.load('../final_model/event/test_event_probabilities.npy') # shape: [samples, leads, lat, lon, hazards]
    
    # We will count how many times any pixel exceeded 50% probability for each hazard per month
    monthly_counts = {m: {'heat': 0, 'cold': 0, 'rainfall': 0, 'cyclone': 0} for m in range(1, 13)}
    
    # Subsample to avoid memory issues (take max prob over space and lead for each sample)
    print("Processing probabilities...")
    max_probs = probs.max(axis=(1, 2, 3)) # shape: [samples, hazards]
    
    threshold = 0.5
    for i, t in enumerate(times):
        month = t.astype('datetime64[M]').astype(int) % 12 + 1
        if max_probs[i, 0] > threshold: monthly_counts[month]['heat'] += 1
        if max_probs[i, 1] > threshold: monthly_counts[month]['cold'] += 1
        if max_probs[i, 2] > threshold: monthly_counts[month]['rainfall'] += 1
        if max_probs[i, 3] > threshold: monthly_counts[month]['cyclone'] += 1

    months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = []
    for m in range(1, 13):
        monthly.append({
            'month': months_labels[m-1],
            'heat': monthly_counts[m]['heat'],
            'cold': monthly_counts[m]['cold'],
            'rainfall': monthly_counts[m]['rainfall'],
            'cyclone': monthly_counts[m]['cyclone']
        })
    print("Monthly computed.")
except Exception as e:
    print("Error with npy files:", e)
    monthly = []

out_data = {
    "eventIntensityEvolution": evolution,
    "monthlyDistribution": monthly
}

with open('dashboard_aggregates.json', 'w') as f:
    json.dump(out_data, f)
print("Saved to dashboard_aggregates.json.")
