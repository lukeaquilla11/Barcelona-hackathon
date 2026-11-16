import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Ensure solution directory is in path
sys.path.insert(0, os.path.abspath('solution'))
import possession_model as pm
import features as feat

print("Pre-computing features once for the sweep...")
F = feat.features()

gt = pd.read_csv('starter/full_match_gt.csv')
graded_mask = ((gt['frame'] >= 11185) & (gt['frame'] <= 90651)) | ((gt['frame'] >= 110808) & (gt['frame'] <= 190197))
gt_labels = gt['label'].to_numpy()

def extract_spells(labels, target_labels={'H', 'A'}):
    s = pd.Series(labels)
    groups = (s != s.shift()).cumsum()
    spells = []
    for _, grp in s.groupby(groups):
        lbl = grp.iloc[0]
        if lbl in target_labels:
            spells.append((lbl, grp.index[0], grp.index[-1]))
    return spells

def compute_spell_f1(gt_labels, pred_labels, iou_thresh=0.5, target_labels={'H', 'A'}):
    gt_sp = extract_spells(gt_labels, target_labels)
    pr_sp = extract_spells(pred_labels, target_labels)
    
    matches = []
    for i, (gl, gs, ge) in enumerate(gt_sp):
        for j, (pl, ps, pe) in enumerate(pr_sp):
            if gl == pl:
                inter = max(0, min(ge, pe) - max(gs, ps) + 1)
                if inter > 0:
                    union = max(ge, pe) - min(gs, ps) + 1
                    iou = inter / union
                    if iou >= iou_thresh:
                        matches.append((iou, i, j))
    matches.sort(reverse=True)
    matched_g, matched_p = set(), set()
    for iou, i, j in matches:
        if i not in matched_g and j not in matched_p:
            matched_g.add(i)
            matched_p.add(j)
            
    tp = len(matched_g)
    prec = tp / len(pr_sp) if pr_sp else 0
    rec = tp / len(gt_sp) if gt_sp else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mean_dur = np.mean([(e - s + 1) / 25.0 for _, s, e in pr_sp]) if pr_sp else 0.0
    return f1, len(pr_sp), mean_dur

# Define grid search space
switch_costs = [16.0, 24.0, 32.0]
dead_costs = [10.0, 15.0, 20.0]
min_spells = [50, 75, 100, 125]

results = []

print(f"\nStarting grid search across {len(switch_costs) * len(dead_costs) * len(min_spells)} configurations...")

for switch_cost, dead_cost, min_spell_frames in itertools.product(switch_costs, dead_costs, min_spells):
    # Run prediction with overridden parameters
    labels = pm.predict(
        F=F, 
        switch_cost=switch_cost, 
        dead_cost=dead_cost, 
        min_spell_frames=min_spell_frames
    )
    
    # Evaluate Macro F1 on graded frames for ['H', 'A', 'D']
    macro_f1 = f1_score(gt_labels[graded_mask], labels[graded_mask], labels=['H', 'A', 'D'], average='macro')
    
    # Evaluate Spell F1 (IoU >= 0.5)
    spell_f1, poss_count, mean_dur = compute_spell_f1(gt_labels, labels, iou_thresh=0.5)
    
    results.append({
        'switch_cost': switch_cost,
        'dead_cost': dead_cost,
        'min_spell': min_spell_frames,
        'macroF1': round(macro_f1, 4),
        'spellF1': round(spell_f1, 4),
        'poss': poss_count,
        'dur': round(mean_dur, 1),
        'score': round(macro_f1 + spell_f1, 4)  # Combined score to rank overall performance
    })

df_results = pd.DataFrame(results).sort_values(by='score', ascending=False)
print("\n=== TOP SWEEP RESULTS ===")
print(df_results.to_string(index=False))

# Export top configuration automatically
best = df_results.iloc[0]
print(f"\nBest Config: switch_cost={best['switch_cost']}, dead_cost={best['dead_cost']}, min_spell={best['min_spell']}")
best_labels = pm.predict(F=F, switch_cost=best['switch_cost'], dead_cost=best['dead_cost'], min_spell_frames=int(best['min_spell']))
pm.write('best_submission.csv', best_labels)
print("Saved best submission to best_submission.csv")