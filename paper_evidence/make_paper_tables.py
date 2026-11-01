"""
paper_evidence/make_paper_tables.py

Regenerates every number planned for the manuscript from saved files only.
Prints source file paths for each number.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_section(title):
    print(f"\n{'='*76}")
    print(f"  {title}")
    print(f"{'='*76}")

def main():
    # =========================================================================
    # TABLE 1: INTERNAL DISCOVERY DATASET & EVALUATION METRICS (WITH CIs)
    # =========================================================================
    print_section("TABLE 1: INTERNAL DISCOVERY DATASET & PRIMARY EVALUATION METRICS")

    manifest_path = 'BrainTumorAI/results/split_manifest.csv'
    print(f"  Source: {manifest_path}")
    df_split = pd.read_csv(manifest_path)
    total = len(df_split)
    splits = df_split['split'].value_counts()
    labels = df_split['label'].value_counts()
    print(f"  Total images: {total}")
    print(f"  Train: {splits.get('train',0)}, Validation: {splits.get('validation',0)}, Test: {splits.get('test',0)}")
    print(f"  Tumor: {labels.get('tumor',0)}, No-Tumor: {labels.get('no_tumor',0)}")

    eval_path = 'BrainTumorAI/results/evaluation/evaluation_metrics.json'
    print(f"\n  Source: {eval_path}")
    eval_metrics = load_json(eval_path)
    models_dict = eval_metrics.get('models', {})
    for model_key, m in models_dict.items():
        acc = m.get('accuracy')
        bacc = m.get('balanced_accuracy')
        sens = m.get('tumor_sensitivity_recall')
        spec = m.get('no_tumor_specificity')
        prec = m.get('tumor_precision')
        f1 = m.get('tumor_f1')
        auc = m.get('roc_auc')
        ap = m.get('average_precision')
        cm = m.get('confusion_matrix')
        ci_dup = m.get('confidence_intervals_duplicate_group_stratified_bootstrap_95', {})
        ci_img = m.get('confidence_intervals_image_stratified_bootstrap_95', {})
        print(f"\n  Model: {model_key} (n={m.get('sample_count')}, threshold={m.get('threshold')})")
        print(f"    Confusion Matrix [no_tumor, tumor]: {cm}")
        print(f"    Accuracy:          {acc:.6f} | Dup-Group 95% CI: [{ci_dup.get('accuracy',{}).get('lower_95',0):.4f}, {ci_dup.get('accuracy',{}).get('upper_95',0):.4f}] | Image 95% CI: [{ci_img.get('accuracy',{}).get('lower_95',0):.4f}, {ci_img.get('accuracy',{}).get('upper_95',0):.4f}]")
        print(f"    Balanced Accuracy: {bacc:.6f} | Dup-Group 95% CI: [{ci_dup.get('balanced_accuracy',{}).get('lower_95',0):.4f}, {ci_dup.get('balanced_accuracy',{}).get('upper_95',0):.4f}]")
        print(f"    Sensitivity:       {sens:.6f} | Dup-Group 95% CI: [{ci_dup.get('tumor_sensitivity_recall',{}).get('lower_95',0):.4f}, {ci_dup.get('tumor_sensitivity_recall',{}).get('upper_95',0):.4f}]")
        print(f"    Specificity:       {spec:.6f} | Dup-Group 95% CI: [{ci_dup.get('no_tumor_specificity',{}).get('lower_95',0):.4f}, {ci_dup.get('no_tumor_specificity',{}).get('upper_95',0):.4f}]")
        print(f"    Precision:         {prec:.6f} | Dup-Group 95% CI: [{ci_dup.get('tumor_precision',{}).get('lower_95',0):.4f}, {ci_dup.get('tumor_precision',{}).get('upper_95',0):.4f}]")
        print(f"    F1:                {f1:.6f} | Dup-Group 95% CI: [{ci_dup.get('tumor_f1',{}).get('lower_95',0):.4f}, {ci_dup.get('tumor_f1',{}).get('upper_95',0):.4f}]")
        print(f"    ROC-AUC:           {auc:.6f} | Dup-Group 95% CI: [{ci_dup.get('roc_auc',{}).get('lower_95',0):.4f}, {ci_dup.get('roc_auc',{}).get('upper_95',0):.4f}]")
        print(f"    PR-AUC (AP):       {ap:.6f} | Dup-Group 95% CI: [{ci_dup.get('average_precision',{}).get('lower_95',0):.4f}, {ci_dup.get('average_precision',{}).get('upper_95',0):.4f}]")

    p1b_5seed_path = 'paper_evidence/phase1b_5seeds_internal_summary.json'
    if os.path.exists(p1b_5seed_path):
        print(f"\n  Phase 1B 5-Seed (seeds 42-46) Internal Summary Source: {p1b_5seed_path}")
        p1b_5 = load_json(p1b_5seed_path)
        for arch, adata in p1b_5.items():
            agg = adata['aggregate']
            print(f"    {arch} (n={adata['n_seeds']}, seeds={adata['seeds']}): "
                  f"Acc={agg['accuracy']['mean_sd_str']} (95% t-CI: {agg['accuracy']['ci_95_t_str']}), "
                  f"BalAcc={agg['balanced_accuracy']['mean_sd_str']} (95% t-CI: {agg['balanced_accuracy']['ci_95_t_str']}), "
                  f"Sens={agg['sensitivity']['mean_sd_str']}, Spec={agg['specificity']['mean_sd_str']}, "
                  f"AUC={agg['roc_auc']['mean_sd_str']} (95% t-CI: {agg['roc_auc']['ci_95_t_str']})")

    # =========================================================================
    # TABLE 2: CLEAN-TEST RESCORE (EXCLUDING 141 CHENG-MATCHED TEST IMAGES)
    # =========================================================================
    print_section("TABLE 2: CLEAN-TEST EVALUATION (EXCLUDING 141 CHENG-MATCHED IMAGES)")
    clean_path = 'paper_evidence/phase3_clean_test_rescore.json'
    print(f"  Source: {clean_path}")
    if os.path.exists(clean_path):
        clean_data = load_json(clean_path)
        print(f"  Removed Cheng-matched test images: {clean_data['leak_test_images_count']}")
        for mname in ['resnet18', 'small_cnn']:
            for sub in ['full', 'clean']:
                d = clean_data[mname][sub]
                print(f"  {mname} ({sub}, n={d['n']}, tumor={d['n_tumor']}, no_tumor={d['n_no_tumor']}): "
                      f"Acc={d['acc']:.4f}, BalAcc={d['bacc']:.4f}, Sens={d['sens']:.4f}, Spec={d['spec']:.4f}, "
                      f"F1={d['f1']:.4f}, AUC={d['roc_auc']:.4f}, CM={d['cm']}")

    # =========================================================================
    # TABLE 3: BATCH SIZE EFFECTS (DISCOVERY & BraTS 10K)
    # =========================================================================
    print_section("TABLE 3: BATCH SIZE EFFECTS (DISCOVERY TEST N=606 & BraTS N=10,000)")

    bs_path = 'paper_evidence/discovery_batch_size_effects.json'
    print(f"  Source: {bs_path}")
    bs = load_json(bs_path)
    for model_name, data in bs.items():
        print(f"  Discovery {model_name} (batch 1 vs 32, n=606):")
        print(f"    Max abs diff: {data['max_abs_diff']:.8e}, Mean abs diff: {data['mean_abs_diff']:.8e}, Flips: {data['flips']}")
        if data['flipped_cases']:
            for fc in data['flipped_cases']:
                print(f"      Flipped image: {fc['image']}: score_b1={fc['score_b1']:.8f} (pred={fc['pred_b1']}) -> score_b32={fc['score_b32']:.8f} (pred={fc['pred_b32']})")

    b10k_path = 'paper_evidence/brats_10k_batch_size_effects.json'
    print(f"\n  Source: {b10k_path}")
    if os.path.exists(b10k_path):
        b10k = load_json(b10k_path)
        print(f"  BraTS 10k stratified subset (batch 1 vs 128, n={b10k['n_slices']}, patients={b10k['n_patients']}):")
        print(f"    Max abs diff: {b10k['max_abs_diff']:.8e}, Mean abs diff: {b10k['mean_abs_diff']:.8e}, Flips: {b10k['flips']}")
        print(f"    Batch 1:   Acc={b10k['batch_1_metrics']['accuracy']:.6f}, Sens={b10k['batch_1_metrics']['sensitivity']:.6f}, Spec={b10k['batch_1_metrics']['specificity']:.6f}, AUC={b10k['batch_1_metrics']['auc']:.6f}")
        print(f"    Batch 128: Acc={b10k['batch_128_metrics']['accuracy']:.6f}, Sens={b10k['batch_128_metrics']['sensitivity']:.6f}, Spec={b10k['batch_128_metrics']['specificity']:.6f}, AUC={b10k['batch_128_metrics']['auc']:.6f}")
        print(f"    Deltas:    dAcc={b10k['metric_changes']['accuracy_delta']:.6e}, dSens={b10k['metric_changes']['sensitivity_delta']:.6e}, dSpec={b10k['metric_changes']['specificity_delta']:.6e}, dAUC={b10k['metric_changes']['auc_delta']:.6e}")

    # =========================================================================
    # TABLE 4: PATIENT LEAKAGE (DISCOVERY vs CHENG)
    # =========================================================================
    print_section("TABLE 4: PATIENT LEAKAGE COUNTS (DISCOVERY vs CHENG)")

    audit_path = 'BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json'
    leak_path = 'paper_evidence/verified_leakage_claims.json'
    print(f"  Sources: {audit_path}, {leak_path}")
    audit = load_json(audit_path)
    leak = load_json(leak_path)
    matches = audit['matches']['high_similarity_matches']
    print(f"  Exact hash match edges: {leak['matches_and_edges']['exact_pixel_matches']}")
    print(f"  High-similarity (pHash<=4 & SSIM>=0.95) edges: {len(matches)}")

    distinct_disc = len(set(m['local_source_file'] for m in matches))
    distinct_cheng = len(set(m['cheng_slice'] for m in matches))
    all_patients = set(m['patient_id'] for m in matches)
    print(f"  Distinct Discovery files matched: {distinct_disc}")
    print(f"  Distinct Cheng slices matched: {distinct_cheng}")
    print(f"  Total Cheng patients matched: {len(all_patients)} / 233 (unmatched: {233 - len(all_patients)})")

    from collections import defaultdict
    split_patients = defaultdict(set)
    for m in matches:
        split_patients[m['local_split']].add(m['patient_id'])
    train_p = split_patients.get('train', set())
    val_p = split_patients.get('validation', set())
    test_p = split_patients.get('test', set())

    print(f"  Patients by split: Train={len(train_p)}, Validation={len(val_p)}, Test={len(test_p)}")
    print(f"  Patients in all three splits (train & val & test): {len(train_p & val_p & test_p)}")
    print(f"  Test patients also in train: {len(test_p & train_p)}")
    print(f"  Test patients NOT in train: {len(test_p - train_p)} -> {sorted(list(test_p - train_p))}")
    print(f"  Strictly test-only patients: {len(test_p - train_p - val_p)} -> {sorted(list(test_p - train_p - val_p))}")
    print(f"  Strictly validation-only patients: {len(val_p - train_p - test_p)} -> {sorted(list(val_p - train_p - test_p))}")

    test_matches = [m for m in matches if m['local_split'] == 'test']
    test_with_train = [m for m in test_matches if m['patient_id'] in train_p]
    test_without_train = [m for m in test_matches if m['patient_id'] not in train_p]
    print(f"  Discovery test images matched to Cheng: {len(test_matches)} (134 with patient in train, 7 without patient in train)")
    for m in test_without_train:
        p = m['patient_id']
        p_splits = sorted(set(x['local_split'] for x in matches if x['patient_id'] == p))
        print(f"    {os.path.basename(m['local_source_file'])} -> {m['cheng_slice']} (patient {p}, splits: {p_splits})")

    # =========================================================================
    # TABLE 5: CHENG EXTERNAL VALIDATION (UNSCREENED & OVERLAP-SCREENED)
    # =========================================================================
    print_section("TABLE 5: CHENG EXTERNAL VALIDATION")

    for variant, label in [
        ('cheng_figshare_v8', 'Unscreened (3,064 slices, 233 patients)'),
        ('cheng_figshare_v8_overlap_screened', 'Overlap-Screened (1,689 slices, 152 patients)')
    ]:
        metrics_path = f'BrainTumorAI/results/external_validation/{variant}/metrics.json'
        print(f"\n  [{label}] Source: {metrics_path}")
        if os.path.exists(metrics_path):
            md = load_json(metrics_path)
            training_batch = md.get('checkpoint_metadata', {}).get('training_config', {}).get('batch_size', 'N/A')
            print(f"    Checkpoint training_config.batch_size (line 85): {training_batch} (TRAINING batch size)")
            print(f"    Evaluation batch_size in metrics.json: NOT VERIFIED (not recorded in metrics.json; external_validation.py default is 32 at line 801)")
            m = md.get('metrics', {})
            sens = m.get('tumor_sensitivity_recall')
            mean_score = m.get('mean_tumor_score')
            ci = m.get('confidence_intervals_patient_cluster_bootstrap_95', {}).get('tumor_sensitivity_recall', {})
            print(f"    Tumor Sensitivity: {sens:.6f} (95% Patient-Cluster CI: {ci.get('lower_95',0):.4f} - {ci.get('upper_95',0):.4f})")
            print(f"    Mean Tumor Score:  {mean_score:.6f}")
            subtypes = m.get('tumor_subtype_sensitivity_patient_cluster_bootstrap_95', {})
            if subtypes:
                for st_name, st_val in subtypes.items():
                    st_ci = st_val.get('confidence_interval_patient_cluster_bootstrap_95', {})
                    print(f"      Subtype {st_name} (n={st_val['sample_count']}, patients={st_val['patient_count']}): "
                          f"Sens={st_val['tumor_sensitivity_recall']:.4f} (95% CI: {st_ci.get('lower_95',0):.4f} - {st_ci.get('upper_95',0):.4f})")

    # =========================================================================
    # TABLE 6: BraTS ORIENTED (PRIMARY) HEADLINE METRICS, HISTOGRAMS, ROC, ORACLE
    # =========================================================================
    print_section("TABLE 6: BraTS ORIENTED (PRIMARY) FULL REPORT")

    detail_path = 'paper_evidence/brats_oriented_detailed_report.json'
    print(f"  Source: {detail_path}")
    det = load_json(detail_path)
    hm = det['headline_metrics']
    ci = det['bootstrap_ci_95']
    cm = hm['confusion_matrix']
    print(f"  Total slices: {cm['tn']+cm['fp']+cm['fn']+cm['tp']} (1,251 patients)")
    print(f"  Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    for metric in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']:
        val = hm[metric]
        c = ci[metric]
        print(f"  {metric:18s}: {val:.6f} (95% CI: {c['ci_95']} [{c['low']:.6f}, {c['high']:.6f}])")
    comp_path = 'paper_evidence/brats_gpu_comparison_report.json'
    comp = load_json(comp_path)
    pr_ci = comp['cohorts']['brats_oriented']['bootstrap_ci_95']['pr_auc']
    print(f"  {'pr_auc':18s}: {hm['pr_auc']:.6f} (95% CI: {pr_ci[0]:.4f} - {pr_ci[1]:.4f} [{pr_ci[0]:.6f}, {pr_ci[1]:.6f}])")

    print("\n  Score Histograms:")
    for cls in ['tumor', 'no_tumor']:
        h = det['histograms'][cls]
        print(f"    {cls}: mean={h['mean']:.6f}, median={h['median']:.6f}")
        for bin_name, count in h['histogram'].items():
            print(f"      {bin_name}: {count}")

    print("\n  ROC Operating Points:")
    for op in det['roc_operating_points']:
        print(f"    threshold={op['threshold']:.1f}: Sens={op['sensitivity']:.6f}, Spec={op['specificity']:.6f}, Acc={op['accuracy']:.6f}")

    ot = det['oracle_threshold_non_claimable']
    print(f"\n  Oracle Threshold ({ot['status']}):")
    print(f"    Threshold={ot['oracle_threshold']:.8f}, Acc={ot['accuracy']:.6f}, BalAcc={ot['balanced_accuracy']:.6f}, Sens={ot['sensitivity']:.6f}, Spec={ot['specificity']:.6f}, Youden_J={ot['youden_index']:.6f}")

    # =========================================================================
    # TABLE 7: BraTS OLD vs ORIENTED vs FLIP-LR SIDE BY SIDE + PAIRED BOOTSTRAP CIs
    # =========================================================================
    print_section("TABLE 7: BraTS OLD vs ORIENTED vs FLIP-LR + PAIRED BOOTSTRAP CIs")
    supp_path = 'paper_evidence/brats_supplementary_cIs_and_subgroups.json'
    print(f"  Sources: {comp_path}, {supp_path}")
    supp = load_json(supp_path)
    for c_key in ['brats_old', 'brats_oriented', 'brats_oriented_flipLR']:
        cd = comp['cohorts'][c_key]['metrics']
        cb = comp['cohorts'][c_key]['bootstrap_ci_95']
        print(f"\n  Cohort: {c_key} (CM={cd['confusion_matrix']}):")
        for m in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc', 'pr_auc']:
            print(f"    {m:18s}: {cd[m]:.6f} (95% CI: {cb[m][0]:.4f} - {cb[m][1]:.4f} [{cb[m][0]:.6f}, {cb[m][1]:.6f}])")

    print("\n  Paired Patient-Cluster Bootstrap Differences (2,000 resamples, seed 42):")
    for pair_key, pair_data in supp['paired_bootstrap_comparisons'].items():
        print(f"    {pair_key}:")
        for m in ['auc', 'accuracy', 'sensitivity', 'specificity']:
            md = pair_data[m]
            print(f"      Delta {m:12s}: {md['point_diff']:+.6f} (95% CI: [{md['ci_95_low']:+.6f}, {md['ci_95_high']:+.6f}] | {md['ci_95_str']})")

    # =========================================================================
    # TABLE 8: TUMOR-SIZE BINS & PER-PATIENT DETECTION
    # =========================================================================
    print_section("TABLE 8: TUMOR-SIZE BINS (EXPLORATORY + POST-HOC) & PER-PATIENT DETECTION")
    bins_path = 'paper_evidence/brats_tumor_size_bins.json'
    print(f"  Sources: {supp_path}, {bins_path}")
    bins_raw = load_json(bins_path)
    print(f"  Pre-specified tumor voxel quartile boundaries: {bins_raw['quartile_boundaries']}")
    for bname, bdata in supp['tumor_size_bins_with_ci_and_auc'].items():
        print(f"  {bname:38s}: n_slices={bdata['n_slices']:5d}, n_patients={bdata['n_patients']:4d}, "
              f"Sens={bdata['sensitivity']:.6f} (95% CI: {bdata['sensitivity_ci_95']}), "
              f"AUC_vs_neg={bdata['auc_vs_all_negatives']:.6f}")

    ppd = supp['per_patient_detection']
    frac = ppd['fraction_of_tumor_slices_detected']
    print(f"\n  Per-Patient Tumor Detection (n={ppd['total_patients_with_tumor_slices']} patients):")
    print(f"    Patients with >=1 detected tumor slice: {ppd['patients_with_at_least_one_detected_tumor_slice']} ({ppd['patient_level_any_slice_sensitivity']*100:.2f}%)")
    print(f"    Patients with 0 detected tumor slices:  {ppd['patients_with_zero_detected_tumor_slices']} ({(1-ppd['patient_level_any_slice_sensitivity'])*100:.2f}%)")
    print(f"    Fraction of tumor slices detected per patient: min={frac['min']:.6f}, Q25={frac['q25']:.6f}, median={frac['median']:.6f}, mean={frac['mean']:.6f}, Q75={frac['q75']:.6f}, max={frac['max']:.6f}")

    # =========================================================================
    # TABLE 9: SLICE-POSITION TERTILES (EXPLORATORY)
    # =========================================================================
    print_section("TABLE 9: SLICE-POSITION TERTILES (EXPLORATORY)")
    print(f"  Source: {supp_path}")
    for tname, tdata in supp['slice_position_tertiles_with_ci'].items():
        print(f"  {tname:30s}: n={tdata['n_slices']} (tumor={tdata['n_tumor_slices']}, no_tumor={tdata['n_no_tumor_slices']}), "
              f"Sens={tdata['sensitivity']:.6f} ({tdata['sensitivity_ci_95']}), "
              f"Spec={tdata['specificity']:.6f} ({tdata['specificity_ci_95']}), "
              f"AUC={tdata['auc']:.6f} ({tdata['auc_ci_95']})")

    # =========================================================================
    # TABLE 10: OTHER CHECKPOINTS ON 250-PATIENT SUBSET (5 SEEDS + 95% CIs)
    # =========================================================================
    print_section("TABLE 10: OTHER CHECKPOINTS ON 250-PATIENT STRATIFIED SUBSET (5 SEEDS + 95% CIs)")
    ck250_path = 'paper_evidence/brats_250_5seeds_with_cis.json'
    print(f"  Source: {ck250_path}")
    if os.path.exists(ck250_path):
        ck250 = load_json(ck250_path)
        si = ck250['subset_info']
        print(f"  Subset: {si['n_patients']} patients, {si['n_slices']} slices (tumor={si['n_tumor_slices']}, no_tumor={si['n_no_tumor_slices']})")
        print(f"  Bootstrap method: {si['bootstrap_method']}")
        for arch, adata in ck250['architectures'].items():
            print(f"\n  Architecture: {arch.upper()} (Phase 1B seeds={adata['n_phase1b_seeds']})")
            for r in adata['individual_runs']:
                bci = r.get('bootstrap_ci_95', {})
                print(f"    {r['model_name']:28s} ({r['checkpoint']}): "
                      f"Acc={r['accuracy']:.4f} ({bci.get('accuracy',{}).get('ci_95_str','')}), "
                      f"BalAcc={r['balanced_accuracy']:.4f} ({bci.get('balanced_accuracy',{}).get('ci_95_str','')}), "
                      f"Sens={r['sensitivity']:.4f} ({bci.get('sensitivity',{}).get('ci_95_str','')}), "
                      f"Spec={r['specificity']:.4f} ({bci.get('specificity',{}).get('ci_95_str','')}), "
                      f"AUC={r['auc']:.4f} ({bci.get('auc',{}).get('ci_95_str','')})")
            s_all = adata.get('summary_all_phase1b_seeds', {})
            if s_all:
                print(f"    Mean +/- Sample SD (5 seeds 42-46): "
                      f"Acc={s_all['accuracy']['mean_sd_sample_str']} [Boot CI: {s_all['accuracy']['patient_bootstrap_ci_95_str']} | Seed t-CI: {s_all['accuracy']['seed_t_ci_95_str']}], "
                      f"BalAcc={s_all['balanced_accuracy']['mean_sd_sample_str']} [Boot CI: {s_all['balanced_accuracy']['patient_bootstrap_ci_95_str']}], "
                      f"Sens={s_all['sensitivity']['mean_sd_sample_str']} [Boot CI: {s_all['sensitivity']['patient_bootstrap_ci_95_str']}], "
                      f"Spec={s_all['specificity']['mean_sd_sample_str']} [Boot CI: {s_all['specificity']['patient_bootstrap_ci_95_str']}], "
                      f"AUC={s_all['auc']['mean_sd_sample_str']} [Boot CI: {s_all['auc']['patient_bootstrap_ci_95_str']} | Seed t-CI: {s_all['auc']['seed_t_ci_95_str']}]")
            s_43_46 = adata.get('summary_seeds_43_to_46')
            if s_43_46:
                print(f"    Mean +/- Sample SD (4 seeds 43-46): "
                      f"Acc={s_43_46['accuracy']['mean_sd_sample_str']} [Boot CI: {s_43_46['accuracy']['patient_bootstrap_ci_95_str']}], "
                      f"BalAcc={s_43_46['balanced_accuracy']['mean_sd_sample_str']} [Boot CI: {s_43_46['balanced_accuracy']['patient_bootstrap_ci_95_str']}], "
                      f"Sens={s_43_46['sensitivity']['mean_sd_sample_str']} [Boot CI: {s_43_46['sensitivity']['patient_bootstrap_ci_95_str']}], "
                      f"Spec={s_43_46['specificity']['mean_sd_sample_str']} [Boot CI: {s_43_46['specificity']['patient_bootstrap_ci_95_str']}], "
                      f"AUC={s_43_46['auc']['mean_sd_sample_str']} [Boot CI: {s_43_46['auc']['patient_bootstrap_ci_95_str']}]")
        p_arch = ck250.get('paired_architecture_comparisons_5seed_mean', {})
        if p_arch:
            print("\n  Paired Architecture 5-Seed Mean Differences (2,000-resample patient-cluster bootstrap, seed 42):")
            for pk, pv in p_arch.items():
                print(f"    {pk}: dAcc={pv['accuracy']['ci_95_str']}, dBalAcc={pv['balanced_accuracy']['ci_95_str']}, "
                      f"dSens={pv['sensitivity']['ci_95_str']}, dSpec={pv['specificity']['ci_95_str']}, dAUC={pv['auc']['ci_95_str']}")

    # =========================================================================
    # TABLE 11: NATIVE BraTS OVERLAP AUDIT & POSITIVE CONTROLS
    # =========================================================================
    print_section("TABLE 11: NATIVE BraTS OVERLAP AUDIT & POSITIVE CONTROLS")
    nat_path = 'paper_evidence/brats_native_overlap_audit.json'
    pc_path = 'paper_evidence/positive_control_overlap_audit.json'
    print(f"  Sources: {nat_path}, {pc_path}")
    if os.path.exists(nat_path):
        nat = load_json(nat_path)
        meta = nat.get('metadata', {})
        all_aud = nat.get('all_patient_oriented_audit', {})
        rot_aud = nat.get('multi_orientation_30_patient_audit', {})
        print(f"  Native BraTS Overlap Audit:")
        print(f"    Method: {meta.get('method')}")
        print(f"    imagehash_used={meta.get('imagehash_used')}, skimage_used={meta.get('skimage_used')}")
        print(f"    Patients audited: {meta.get('brats_patients_audited')}, Slices audited: {meta.get('brats_slices_audited')}")
        print(f"    Discovery images indexed: {meta.get('discovery_images_indexed')}")
        print(f"    Exact matches: {all_aud.get('exact_matches_count')}, High-similarity matches (pHash<=4 & SSIM>=0.95): {all_aud.get('high_similarity_matches_count')}")
        print(f"    Multi-orientation check (30 patients, angles {rot_aud.get('angles_tested')}): hits={rot_aud.get('hits_count')}")
        print(f"    Runtime seconds (from task-2070 log): 2255.8s for 1,251 patients (169,066 slices)")
    if os.path.exists(pc_path):
        pc = load_json(pc_path)
        pc1 = pc['positive_control_1_cheng_discovery_50_pairs']
        pc2 = pc['positive_control_2_brats_50_injected_slices']
        print(f"  Positive Control 1 (Cheng-Discovery 50 known duplicate pairs): {pc1['pairs_detected']}/{pc1['pairs_tested']} detected ({pc1['detection_rate_percent']}%), exact expected file matched={pc1['exact_expected_discovery_file_matched']}/50")
        print(f"  Positive Control 2 (50 injected BraTS slices): {pc2['slices_detected']}/{pc2['slices_injected']} detected ({pc2['detection_rate_percent']}%), Hamming={pc2['hamming_distance_all']}, SSIM={pc2['ssim_min_mean_max']}")
        print(f"  Positive control runtime: {pc['runtime_seconds']:.2f}s")

    # =========================================================================
    # TABLE 12: LEAKAGE-FREE BASELINES (PATIENT-SPLIT BraTS, DECONTAMINATED DISCOVERY, CHENG)
    # =========================================================================
    print_section("TABLE 12: LEAKAGE-FREE BASELINES (PATIENT-SPLIT BraTS, DISCOVERY, & CHENG)")
    lfb_path = 'paper_evidence/leakage_free_baselines.json'
    print(f"  Source: {lfb_path}")
    if os.path.exists(lfb_path):
        lfb = load_json(lfb_path)
        b1 = lfb['baseline_1_brats_patient_split']
        sc = b1['split_counts']
        print(f"  1. Patient-Split BraTS (brats_oriented): Train={sc['train_patients']} patients ({sc['train_slices_sampled']} slices), "
              f"Val={sc['val_patients']} patients ({sc['val_slices_sampled']} slices), Held-out Test={sc['test_patients']} patients ({sc['test_slices_full']} slices)")
        m_b1 = b1['models_on_brats_250_test']
        zs = m_b1['zeroshot_discovery_resnet18_reference']
        print(f"    Zero-Shot Discovery ResNet-18 (Ref): Acc={zs['accuracy']:.4f} ({zs['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
              f"BalAcc={zs['balanced_accuracy']:.4f}, Sens={zs['sensitivity']:.4f}, Spec={zs['specificity']:.4f}, "
              f"AUC={zs['auc']:.4f} ({zs['bootstrap_ci_95']['auc']['ci_95_str']}), PR-AUC={zs['pr_auc']:.4f}, CM={zs['confusion_matrix']}")
        for k_mod in ['resnet18_brats_patient_split_from_imagenet', 'resnet18_brats_finetuned_from_discovery', 'small_cnn_brats_patient_split_from_scratch']:
            tm = m_b1[k_mod]['test_metrics']
            print(f"    {k_mod:42s}: Acc={tm['accuracy']:.4f} ({tm['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
                  f"BalAcc={tm['balanced_accuracy']:.4f} ({tm['bootstrap_ci_95']['balanced_accuracy']['ci_95_str']}), "
                  f"Sens={tm['sensitivity']:.4f} ({tm['bootstrap_ci_95']['sensitivity']['ci_95_str']}), "
                  f"Spec={tm['specificity']:.4f} ({tm['bootstrap_ci_95']['specificity']['ci_95_str']}), "
                  f"AUC={tm['auc']:.4f} ({tm['bootstrap_ci_95']['auc']['ci_95_str']}), PR-AUC={tm['pr_auc']:.4f}, CM={tm['confusion_matrix']}")
        print("    Paired Differences vs Zero-Shot Discovery ResNet-18:")
        for pk, pv in b1['paired_bootstrap_comparisons_vs_zeroshot'].items():
            print(f"      {pk}: dAcc={pv['accuracy']['ci_95_str']}, dSens={pv['sensitivity']['ci_95_str']}, dSpec={pv['specificity']['ci_95_str']}, dAUC={pv['auc']['ci_95_str']}")

        b2 = lfb['baseline_2_discovery_patient_decontaminated']
        print(f"\n  2. Patient-Decontaminated Discovery ResNet-18 (removed {b2['removed_train_val_slices']} leaked train/val slices; clean_train={b2['clean_train_count']}, clean_val={b2['clean_val_count']}):")
        for sub_k in ['test_full_606', 'test_clean_465', 'test_cheng_matched_141_now_unseen_patients']:
            sd = b2[sub_k]
            print(f"    {sub_k:42s} (n={sd['n']}): Acc={sd['accuracy']:.4f}, Sens={sd['sensitivity']:.4f}, Spec={sd['specificity']}, AUC={sd['roc_auc']}, CM={sd['confusion_matrix']}")

        print("\n    External Evaluation on Cheng:")
        for ck, cv in b1['external_evaluation_on_cheng'].items():
            print(f"      {ck:50s} (n={cv['n_slices']}, patients={cv['n_patients']}): Sens={cv['sensitivity']:.4f} (95% CI: {cv['ci_95_str']}), MeanScore={cv['mean_score']:.4f}")

        b3 = lfb['baseline_3_cheng_3class_patient_vs_slice_split']
        print("\n  3. Cheng 3-Class Subtype ResNet-18: Strict Patient-Split vs Random Slice-Split:")
        for sp_k in ['strict_patient_split', 'random_slice_split_leaked']:
            sp = b3[sp_k]
            bci = sp['bootstrap_ci_95']
            print(f"    {sp['split_type']:48s} (train={sp['n_train_slices']}, val={sp['n_val_slices']}, test={sp['n_test_slices']} slices, test_patients={sp['n_test_patients']}): "
                  f"Acc={sp['accuracy']:.4f} ({bci['accuracy']['ci_95_str']}), "
                  f"Macro-BalAcc={sp['macro_balanced_accuracy']:.4f} ({bci['macro_balanced_accuracy']['ci_95_str']}), "
                  f"Macro-F1={sp['macro_f1']:.4f} ({bci['macro_f1']['ci_95_str']}), "
                  f"Macro-AUC={sp['macro_roc_auc_ovr']:.4f} ({bci['macro_roc_auc_ovr']['ci_95_str']}), CM={sp['confusion_matrix']}")
        infl = b3['leakage_optimism_delta']
        print(f"    Slice-Split Leakage Optimism Delta: dAcc={infl['accuracy_inflation']:+.4f}, dMacroBalAcc={infl['macro_balanced_accuracy_inflation']:+.4f}, "
              f"dMacroF1={infl['macro_f1_inflation']:+.4f}, dMacroAUC={infl['macro_auc_inflation']:+.4f}")

    # =========================================================================
    # TABLE 13: QUANTITATIVE GRAD-CAM EVALUATION ON BraTS TUMOR MASKS
    # =========================================================================
    print_section("TABLE 13: QUANTITATIVE GRAD-CAM EVALUATION ON BraTS TUMOR SEGMENTATION MASKS")
    gc_path = 'paper_evidence/gradcam_brats_evaluation.json'
    print(f"  Source: {gc_path}")
    if os.path.exists(gc_path):
        gc_data = load_json(gc_path)
        print(f"  Cohort: {gc_data['cohort']} (n_tumor_slices={gc_data['n_tumor_slices']}, n_patients={gc_data['n_patients']})")
        print(f"  Layer: {gc_data['gradcam_layer']}, Batched vs Single verification max diff: {gc_data['batched_vs_single_verification_max_abs_diff']:.2e}")
        for m_key, m_val in gc_data['models'].items():
            ov = m_val['overall_all_tumor_slices']['metrics']
            tp = m_val['by_detection_status']['detected_true_positives_score_gte_05']
            fn = m_val['by_detection_status']['missed_false_negatives_score_lt_05']
            print(f"\n  Model: {m_key} ({m_val['checkpoint']})")
            print(f"    Overall All Tumor Slices (n={m_val['overall_all_tumor_slices']['n_slices']}):")
            print(f"      Pointing-Game (exact): {ov['pointing_hit_exact']['mean']:.4f} ({ov['pointing_hit_exact']['ci_95_str']})")
            print(f"      Pointing-Game (15px):  {ov['pointing_hit_15px']['mean']:.4f} ({ov['pointing_hit_15px']['ci_95_str']})")
            print(f"      EBPG (tumor energy):   {ov['ebpg_tumor']['mean']:.4f} ({ov['ebpg_tumor']['ci_95_str']}) vs Null image area={ov['null_tumor_area_frac_image']['mean']:.4f} (Enrichment={ov['enrichment_vs_image']['mean']:.2f}x)")
            print(f"      EBPG (brain energy):   {ov['ebpg_brain']['mean']:.4f} ({ov['ebpg_brain']['ci_95_str']})")
            print(f"      Dice@0.5:              {ov['dice_05']['mean']:.4f} ({ov['dice_05']['ci_95_str']}), IoU@0.5: {ov['iou_05']['mean']:.4f} ({ov['iou_05']['ci_95_str']})")
            print(f"      Dice@0.7:              {ov['dice_07']['mean']:.4f} ({ov['dice_07']['ci_95_str']}), IoU@0.7: {ov['iou_07']['mean']:.4f} ({ov['iou_07']['ci_95_str']})")
            print(f"      Dice@TopK:             {ov['dice_topk']['mean']:.4f} ({ov['dice_topk']['ci_95_str']}), IoU@TopK: {ov['iou_topk']['mean']:.4f} ({ov['iou_topk']['ci_95_str']})")
            print(f"      Soft Dice:             {ov['soft_dice']['mean']:.4f} ({ov['soft_dice']['ci_95_str']})")
            print(f"    Detected True Positives (n={tp['n_slices']}): Pointing(exact)={tp['metrics']['pointing_hit_exact']['mean']:.4f} ({tp['metrics']['pointing_hit_exact']['ci_95_str']}), "
                  f"Pointing(15px)={tp['metrics']['pointing_hit_15px']['mean']:.4f} ({tp['metrics']['pointing_hit_15px']['ci_95_str']}), "
                  f"EBPG={tp['metrics']['ebpg_tumor']['mean']:.4f} ({tp['metrics']['ebpg_tumor']['ci_95_str']}), "
                  f"Dice@0.5={tp['metrics']['dice_05']['mean']:.4f} ({tp['metrics']['dice_05']['ci_95_str']}), Dice@TopK={tp['metrics']['dice_topk']['mean']:.4f}")
            print(f"    Missed False Negatives  (n={fn['n_slices']}): Pointing(exact)={fn['metrics']['pointing_hit_exact']['mean']:.4f} ({fn['metrics']['pointing_hit_exact']['ci_95_str']}), "
                  f"Pointing(15px)={fn['metrics']['pointing_hit_15px']['mean']:.4f} ({fn['metrics']['pointing_hit_15px']['ci_95_str']}), "
                  f"EBPG={fn['metrics']['ebpg_tumor']['mean']:.4f} ({fn['metrics']['ebpg_tumor']['ci_95_str']}), "
                  f"Dice@0.5={fn['metrics']['dice_05']['mean']:.4f} ({fn['metrics']['dice_05']['ci_95_str']}), Dice@TopK={fn['metrics']['dice_topk']['mean']:.4f}")
        print("\n  Paired Bootstrap Grad-CAM Comparisons vs Zero-Shot Discovery:")
        for pk, pv in gc_data.get('paired_bootstrap_comparisons', {}).items():
            print(f"    {pk}: dPointing(exact)={pv['pointing_hit_exact']['ci_95_str']}, dPointing(15px)={pv['pointing_hit_15px']['ci_95_str']}, "
                  f"dEBPG(tumor)={pv['ebpg_tumor']['ci_95_str']}, dDice@0.5={pv['dice_05']['ci_95_str']}, dDice@TopK={pv['dice_topk']['ci_95_str']}")

    # =========================================================================
    # TABLE 14: MULTI-SEED (SEEDS 42-46) PATIENT-SPLIT BraTS MODELS & GRAD-CAM
    # =========================================================================
    print_section("TABLE 14: MULTI-SEED (SEEDS 42-46) PATIENT-SPLIT BraTS MODELS & GRAD-CAM")
    ms_path = 'paper_evidence/leakage_free_multiseed.json'
    gc_ms_path = 'paper_evidence/gradcam_brats_multiseed.json'
    print(f"  Sources: {ms_path}, {gc_ms_path}")
    if os.path.exists(ms_path):
        ms = load_json(ms_path)
        print(f"  Manifest: {ms['manifest_path']} (SHA256={ms['manifest_sha256']})")
        print(f"  Predictions: {ms['predictions_path']} (SHA256={ms['predictions_sha256']})")
        print(f"  Eval batch size (BraTS): {ms['eval_batch_size_brats']}, Eval batch size (Cheng): {ms['eval_batch_size_cheng']}, Threshold: {ms['threshold']}")

        print("\n  (a) 15-Checkpoint Results on Held-Out BraTS-250 (33,696 slices, 250 patients) with 95% Patient-Cluster Bootstrap CIs:")
        for mkey, mdata in ms['models'].items():
            print(f"\n    Model {mdata['model_code']}: {mkey} (arch={mdata['architecture']}, init={mdata['initialization']}, lr={mdata['learning_rate']})")
            for s_str, sd in mdata['per_seed'].items():
                bci = sd['bootstrap_ci_95']
                print(f"      Seed {s_str} (epoch={sd['selected_epoch']}, val_bal_acc={sd['validation_balanced_accuracy']:.4f}, val_loss={sd['validation_loss']:.4f}, CM={sd['confusion_matrix']}):")
                print(f"        Acc={sd['accuracy']:.4f} ({bci['accuracy']['ci_95_str']}), BalAcc={sd['balanced_accuracy']:.4f} ({bci['balanced_accuracy']['ci_95_str']}), "
                      f"Sens={sd['sensitivity']:.4f} ({bci['sensitivity']['ci_95_str']}), Spec={sd['specificity']:.4f} ({bci['specificity']['ci_95_str']})")
                print(f"        Prec={sd['precision']:.4f} ({bci['precision']['ci_95_str']}), F1={sd['f1']:.4f} ({bci['f1']['ci_95_str']}), "
                      f"ROC-AUC={sd['roc_auc']:.4f} ({bci['roc_auc']['ci_95_str']}), PR-AUC={sd['pr_auc']:.4f} ({bci['pr_auc']['ci_95_str']})")

        print("\n  (b) Per-Model 5-Seed Summary (mean +/- sample SD ddof=1, min, max, 95% t-interval df=4, and 95% Patient-Cluster Bootstrap CI):")
        for mkey, mdata in ms['models'].items():
            agg = mdata['aggregate_5seed']
            print(f"\n    Model {mdata['model_code']}: {mkey}")
            for mn in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc', 'pr_auc']:
                am = agg[mn]
                print(f"      {mn:18s}: {am['mean_sd_str']} | min={am['min']:.4f}, max={am['max']:.4f} | 95% t-CI(df=4): {am['t_ci_95_str']} | Boot 95% CI: {am['patient_bootstrap_ci_95_str']}")

        print("\n  (c) Paired Patient-Cluster Bootstrap Differences (2,000 resamples, seed 42) & Seed-Wise A - B:")
        print("    1. 5-Seed Ensemble Mean Prediction Differences (mean predicted score across 5 seeds):")
        for pk, pv in ms['paired_bootstrap_differences_5seed_ensemble_score'].items():
            print(f"      {pk}: dAcc={pv['accuracy']['ci_95_str']}, dBalAcc={pv['balanced_accuracy']['ci_95_str']}, "
                  f"dSens={pv['sensitivity']['ci_95_str']}, dSpec={pv['specificity']['ci_95_str']}, "
                  f"dF1={pv['f1']['ci_95_str']}, dROC-AUC={pv['roc_auc']['ci_95_str']}, dPR-AUC={pv['pr_auc']['ci_95_str']}")
        print("    2. 5-Seed Mean Metric Differences (average of 5 seed metrics per bootstrap replicate):")
        for pk, pv in ms['paired_bootstrap_differences_5seed_mean_metric'].items():
            print(f"      {pk}: dAcc={pv['accuracy']['ci_95_str']}, dBalAcc={pv['balanced_accuracy']['ci_95_str']}, "
                  f"dSens={pv['sensitivity']['ci_95_str']}, dSpec={pv['specificity']['ci_95_str']}, "
                  f"dF1={pv['f1']['ci_95_str']}, dROC-AUC={pv['roc_auc']['ci_95_str']}, dPR-AUC={pv['pr_auc']['ci_95_str']}")
        print("    3. Seed-Wise Paired Differences (Model A minus Model B across seeds 42-46):")
        for mn, sv in ms['seedwise_paired_differences_A_minus_B'].items():
            vals_str = ", ".join(f"s{k}={v:+.4f}" for k, v in sv['five_seed_values'].items())
            print(f"      {mn:38s}: {sv['mean_sd_str']} (95% t-CI: {sv['t_ci_95_str']}) | [{vals_str}]")

        dom = ms['roc_auc_dominance_check']
        min_e = dom['minimum_roc_auc_among_15_checkpoints']
        max_e = dom['maximum_roc_auc_among_15_checkpoints']
        print("\n    4. ROC-AUC Dominance Check Across All 15 Checkpoints:")
        print(f"      All 15 checkpoints > Zero-Shot Discovery ResNet-18 ({dom['zero_shot_discovery_resnet18_seed42_auc']:.4f})? "
              f"{dom['all_15_checkpoints_above_zeroshot_resnet18_07088']} ({dom['n_checkpoints_above_07088']}/15)")
        print(f"      All 15 checkpoints > Best Discovery Seed DenseNet-121 seed 42 ({dom['best_discovery_densenet121_seed42_auc']:.4f})? "
              f"{dom['all_15_checkpoints_above_best_discovery_densenet121_07255']} ({dom['n_checkpoints_above_07255']}/15)")
        print(f"      Minimum ROC-AUC among 15 checkpoints: {min_e['roc_auc']:.6f} ({min_e['model']} seed {min_e['seed']})")
        print(f"      Maximum ROC-AUC among 15 checkpoints: {max_e['roc_auc']:.6f} ({max_e['model']} seed {max_e['seed']})")

        print("\n  (d) External Evaluation on Cheng Unscreened (3,064 slices, 233 patients) & Overlap-Screened (1,689 slices, 152 patients):")
        for mkey, mdata in ms['models'].items():
            agg = mdata['aggregate_5seed']
            u_agg = agg['cheng_unscreened_3064_sensitivity']
            s_agg = agg['cheng_overlap_screened_1689_sensitivity']
            print(f"\n    Model {mdata['model_code']}: {mkey}")
            for s_str, sd in mdata['per_seed'].items():
                cu = sd['cheng_unscreened_3064']
                cs = sd['cheng_overlap_screened_1689']
                print(f"      Seed {s_str}: Unscreened Sens={cu['sensitivity']:.4f} (95% CI: {cu['ci_95_str']}, mean_score={cu['mean_score']:.4f}) | "
                      f"Overlap-Screened Sens={cs['sensitivity']:.4f} (95% CI: {cs['ci_95_str']}, mean_score={cs['mean_score']:.4f})")
            print(f"      5-Seed Summary Unscreened (3,064):       {u_agg['mean_sd_str']} (min={u_agg['min']:.4f}, max={u_agg['max']:.4f}, 95% t-CI: {u_agg['t_ci_95_str']})")
            print(f"      5-Seed Summary Overlap-Screened (1,689): {s_agg['mean_sd_str']} (min={s_agg['min']:.4f}, max={s_agg['max']:.4f}, 95% t-CI: {s_agg['t_ci_95_str']})")

    if os.path.exists(gc_ms_path):
        gc_ms = load_json(gc_ms_path)
        print("\n  (e) Multi-Seed Grad-CAM Evaluation on BraTS-250 Tumor Slices (n=16,074 slices, 250 patients, seeds 42-46):")
        defs = gc_ms['metric_definitions_from_code']
        print(f"    Code Definitions (from {defs['source_file']}):")
        print(f"      - 15px Pointing Rule (lines {defs['pointing_rule_15px']['lines']}): {defs['pointing_rule_15px']['code']}")
        print(f"      - EBPG Tumor Energy (lines {defs['ebpg_tumor_energy']['lines']}): {defs['ebpg_tumor_energy']['code']}")
        print(f"      - TopK Threshold K  (lines {defs['topk_threshold_K']['lines']}): {defs['topk_threshold_K']['code']}")
        for mkey, gdata in gc_ms['models_multiseed'].items():
            print(f"\n    Grad-CAM Model: {mkey}")
            for s_str, sdata in gdata['per_seed'].items():
                om = sdata['overall_all_tumor_slices']['metrics']
                tp_n = sdata['by_detection_status']['detected_true_positives_score_gte_05']['n_slices']
                fn_n = sdata['by_detection_status']['missed_false_negatives_score_lt_05']['n_slices']
                print(f"      Seed {s_str} (TP={tp_n}, FN={fn_n}): "
                      f"Point(exact)={om['pointing_hit_exact']['mean']:.4f} ({om['pointing_hit_exact']['ci_95_str']}), "
                      f"Point(15px)={om['pointing_hit_15px']['mean']:.4f} ({om['pointing_hit_15px']['ci_95_str']}), "
                      f"EBPG={om['ebpg_tumor']['mean']:.4f} ({om['ebpg_tumor']['ci_95_str']}), "
                      f"Dice@0.5={om['dice_05']['mean']:.4f} ({om['dice_05']['ci_95_str']}), "
                      f"Dice@TopK={om['dice_topk']['mean']:.4f} ({om['dice_topk']['ci_95_str']})")
            agg_g = gdata['aggregate_5seed_overall']
            print(f"      5-Seed Mean +/- SD (95% t-CI df=4):")
            for gk in ['pointing_hit_exact', 'pointing_hit_15px', 'ebpg_tumor', 'dice_05', 'dice_topk']:
                gm = agg_g[gk]
                print(f"        {gk:20s}: {gm['mean_sd_str']} (min={gm['min']:.4f}, max={gm['max']:.4f}, 95% t-CI: {gm['t_ci_95_str']})")

    # =========================================================================
    # TABLE 15: SHA-256 HASHES OF PREDICTION FILES
    # =========================================================================
    print_section("TABLE 15: SHA-256 HASHES OF ALL SAVED PREDICTION FILES")
    pred_files = [
        'paper_evidence/brats_oriented_predictions.csv.gz',
        'paper_evidence/brats_old_predictions.csv.gz',
        'paper_evidence/brats_oriented_flipLR_predictions.csv.gz',
        'paper_evidence/brats_250_other_checkpoints_predictions.csv.gz',
        'paper_evidence/brats_250_leakage_free_baseline_predictions.csv.gz',
        'paper_evidence/brats_250_leakage_free_multiseed_predictions.csv.gz',
        'BrainTumorAI/results/evaluation/test_predictions.csv',
        'BrainTumorAI/results/external_validation/cheng_figshare_v8/predictions.csv',
        'BrainTumorAI/results/external_validation/cheng_figshare_v8_overlap_screened/predictions.csv'
    ]
    for pf in pred_files:
        if os.path.exists(pf):
            print(f"  {pf}: {sha256(pf)}")

    print(f"\n{'='*76}")
    print(f"  END OF PAPER TABLES")
    print(f"{'='*76}")

if __name__ == '__main__':
    main()
