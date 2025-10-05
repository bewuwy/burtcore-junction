#!/usr/bin/env python3
"""
Fill missing 'extremism' labels using regression/classification trained on primary labels.

- Auto-detects primary feature columns (categorical + numeric) and ignores free-form text columns
- Trains on rows where 'extremism' is provided (0/1, or strings like yes/no/true/false)
- Predicts an extremism score for all rows, and fills missing labels using a threshold
- Optionally match prevalence of labeled set when choosing the threshold for unlabeled data
- Supports grouping by an id column and averaging primary label scores across annotators

Usage examples:
  python backend/fill_extremism_from_primary.py \
    --csv labeling/dataset.csv \
    --extremism_column extremism \
    --output labeling/dataset_extremism_filled.csv

  # Dry-run to inspect auto-detected columns and cross-val metrics
  python backend/fill_extremism_from_primary.py --csv labeling/dataset.csv --dry_run

  # Write in place, using logistic and prevalence-matching threshold, grouping by id
  python backend/fill_extremism_from_primary.py --csv labeling/dataset.csv --inplace --match_prevalence --group_by_id
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_IGNORE_BY_NAME = {
    'text', 'Text', 'message', 'Message', 'content', 'Content', 'tweet', 'Tweet',
    'transcript', 'Transcript', 'title', 'Title', 'body', 'Body', 'raw_text', 'RawText',
    'id', 'Id', 'ID', 'uuid', 'UUID', 'video_id', 'VideoId', 'video', 'Video'
}
DEFAULT_ANNOTATOR_TOKENS = {
    'annotator', 'rater', 'worker', 'labeler', 'coder', 'judge', 'grader', 'mturk', 'turk', 'user', 'reviewer'
}


def to_int_label(x) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in {'1', 'true', 'yes', 'y', 'extremist'}:
            return 1
        if xl in {'0', 'false', 'no', 'n', 'not extremist', 'non-extremist', 'non extremist'}:
            return 0
    try:
        val = int(float(x))
        return 1 if val == 1 else 0
    except Exception:
        return None


def is_annotator_column(name: str) -> bool:
    nl = name.lower()
    return any(tok in nl for tok in DEFAULT_ANNOTATOR_TOKENS)


def detect_feature_columns(df: pd.DataFrame, extremism_col: str, explicit_cols: Optional[List[str]], ignore_by_name: set[str], drop_categorical_when_grouping: bool) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) to use as features.
    - Excludes the extremism column
    - If explicit_cols provided, use intersection of provided and df columns
    - Otherwise, auto-detect: keep numeric dtypes and object/category columns with limited cardinality
    - Exclude columns that look like annotator identifiers or metadata
    - Optionally drop categorical columns entirely if grouping by id and averaging
    """
    cols = [c for c in df.columns if c != extremism_col]
    if explicit_cols:
        cols = [c for c in explicit_cols if c in df.columns and c != extremism_col]

    candidate_cols: List[str] = []
    for c in cols:
        if c in ignore_by_name or is_annotator_column(c):
            continue
        candidate_cols.append(c)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    n_rows = len(df)
    for c in candidate_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            if drop_categorical_when_grouping:
                # skip categorical if we're going to average across annotators
                continue
            # object/category: treat as categorical if cardinality is reasonable
            nunique = s.nunique(dropna=True)
            unique_ratio = nunique / max(1, n_rows)
            avg_len = (s.dropna().astype(str).str.len().mean()) if nunique > 0 else 0.0
            # Heuristics: drop likely free-text or id-like columns
            if nunique <= 1000 and unique_ratio <= 0.2 and avg_len <= 50:
                categorical_cols.append(c)
            else:
                # skip
                pass

    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str], model_type: str) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(('num', Pipeline(steps=[
            ('imp', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', Pipeline(steps=[
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),  # drop 'sparse' arg for compatibility
        ]), categorical_cols))

    pre = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)

    if model_type == 'logistic':
        model = LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            C=1.0,
            max_iter=500,
            class_weight='balanced',
            n_jobs=None,
        )
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    return pipe


def choose_threshold(scores: np.ndarray, labeled_y_bin: np.ndarray, match_prevalence: bool, default_threshold: float) -> float:
    if not match_prevalence:
        return float(default_threshold)
    # Match prevalence of positives in labeled set
    pos_rate = float(np.mean(labeled_y_bin)) if labeled_y_bin.size > 0 else 0.5
    if pos_rate <= 0.0:
        return 0.99
    if pos_rate >= 1.0:
        return 0.01
    # threshold s.t. share of scores >= thr is approx pos_rate
    thr = float(np.quantile(scores, 1.0 - pos_rate))
    return thr


def find_id_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    # Try common id patterns
    for cand in ['id', 'Id', 'ID', 'video_id', 'videoId', 'sample_id', 'uid', 'UUID', 'uuid']:
        if cand in df.columns:
            return cand
    # Fallback: first column ending with _id (case-insensitive)
    for c in df.columns:
        if c.lower().endswith('_id'):
            return c
    raise ValueError("Could not find an id column. Please provide --id_column explicitly.")


def main():
    ap = argparse.ArgumentParser(description='Fill extremism labels from primary columns via regression/classification')
    ap.add_argument('--csv', required=True, help='Path to dataset CSV')
    ap.add_argument('--extremism_column', default='extremism', help='Name of extremism column')
    ap.add_argument('--primary_columns', default=None, help='Comma-separated list of columns to use as features; auto-detect if omitted')
    ap.add_argument('--ignore_columns', default=None, help='Comma-separated list of columns to ignore in addition to defaults')
    ap.add_argument('--model_type', choices=['logistic', 'ridge'], default='logistic', help='Model to train')
    ap.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting scores to binary')
    ap.add_argument('--match_prevalence', action='store_true', help='Choose threshold to match labeled positive rate')
    ap.add_argument('--output', default=None, help='Output CSV path; defaults to <input> with _extremism_filled suffix')
    ap.add_argument('--inplace', action='store_true', help='Write back to the same file (overwrites)')
    ap.add_argument('--dry_run', action='store_true', help='Do not write; only report selected columns and metrics')
    ap.add_argument('--limit_rows', type=int, default=None, help='Optional limit of rows to load for faster testing')
    ap.add_argument('--group_by_id', action='store_true', help='Group by id and average numeric primary labels across annotators')
    ap.add_argument('--id_column', default=None, help='Name of the id column to group by (auto-detect if omitted)')
    ap.add_argument('--drop_categorical_when_grouping', action='store_true', help='Drop categorical features when grouping (recommended)')
    ap.add_argument('--exclude_annotator_columns', action='store_true', help='Exclude annotator-related columns like worker_id, annotator, rater, etc.')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, nrows=args.limit_rows)

    if args.extremism_column not in df.columns:
        raise ValueError(f"Extremism column '{args.extremism_column}' not found. Available: {list(df.columns)[:50]}...")

    # Determine id column if grouping is requested
    id_col: Optional[str] = None
    if args.group_by_id:
        id_col = find_id_column(df, args.id_column)
        print(f"Grouping by id column: {id_col}")

    # Build ignore lists
    extra_ign = set([c.strip() for c in args.ignore_columns.split(',')]) if args.ignore_columns else set()
    ignore_by_name = DEFAULT_IGNORE_BY_NAME | extra_ign | {args.extremism_column}
    if args.exclude_annotator_columns:
        # We'll filter annotator-like columns dynamically during detection
        pass

    # Labeled mask at row-level
    labeled_mask_row = df[args.extremism_column].notna()

    # If grouping by id: aggregate numeric features by mean, and the target by mean as well (across annotators)
    if args.group_by_id and id_col:
        # Determine feature columns first on the raw df; but if dropping categorical when grouping, they'll be excluded anyway
        explicit_cols = [c.strip() for c in args.primary_columns.split(',')] if args.primary_columns else None
        numeric_cols, categorical_cols = detect_feature_columns(
            df,
            args.extremism_column,
            explicit_cols,
            ignore_by_name,
            drop_categorical_when_grouping=args.drop_categorical_when_grouping or True,
        )
        # Remove id column from features if accidentally included
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
        if id_col in categorical_cols:
            categorical_cols.remove(id_col)

        # We'll only use numeric columns when grouping by averages
        feature_cols = numeric_cols
        if not feature_cols:
            raise RuntimeError("No numeric feature columns found for grouping. Provide --primary_columns explicitly.")

        # Prepare target as mean of binary extremism per id
        df['_extremism_bin'] = df[args.extremism_column].map(to_int_label)
        # Aggregate
        agg_map = {c: 'mean' for c in feature_cols}
        agg_map['_extremism_bin'] = 'mean'
        df_group = df[[id_col] + feature_cols + ['_extremism_bin']].groupby(id_col, as_index=False).agg(agg_map)
        # Rename target
        df_group = df_group.rename(columns={'_extremism_bin': '_extremism_mean'})

        # Determine which groups are labeled (at least one row labeled -> mean is not NaN)
        labeled_mask_group = df_group['_extremism_mean'].notna()
        y_float = df_group.loc[labeled_mask_group, '_extremism_mean'].to_numpy(dtype=float)
        y_bin = (y_float >= 0.5).astype(int)
        X_labeled = df_group.loc[labeled_mask_group, feature_cols]

        print(f"Groups: {len(df_group)}, labeled groups: {int(labeled_mask_group.sum())} ({labeled_mask_group.mean():.1%})")

        # Build pipeline
        pipe = build_pipeline(feature_cols, [], args.model_type)

        # Cross-validated metrics
        if args.model_type == 'logistic':
            # Stratified on binary labels
            try:
                skf = StratifiedKFold(n_splits=min(5, max(2, np.bincount(y_bin).min())), shuffle=True, random_state=42)
            except Exception:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            oof_scores = np.zeros_like(y_bin, dtype=float)
            fold = 0
            for tr_idx, va_idx in skf.split(np.zeros_like(y_bin), y_bin):
                fold += 1
                pipe_fold = build_pipeline(feature_cols, [], args.model_type)
                pipe_fold.fit(X_labeled.iloc[tr_idx], y_bin[tr_idx])
                s = pipe_fold.predict_proba(X_labeled.iloc[va_idx])[:, 1]
                oof_scores[va_idx] = s
                try:
                    auc = roc_auc_score(y_bin[va_idx], s)
                    ap = average_precision_score(y_bin[va_idx], s)
                    print(f"  Fold {fold}: ROC-AUC={auc:.4f}, AP={ap:.4f}")
                except Exception:
                    pass
            try:
                auc = roc_auc_score(y_bin, oof_scores)
                ap = average_precision_score(y_bin, oof_scores)
                print(f"OOF: ROC-AUC={auc:.4f}, AP={ap:.4f}")
            except Exception:
                pass
        else:
            # Regression CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            oof = np.zeros_like(y_float, dtype=float)
            fold = 0
            for tr_idx, va_idx in kf.split(X_labeled):
                fold += 1
                pipe_fold = build_pipeline(feature_cols, [], args.model_type)
                pipe_fold.fit(X_labeled.iloc[tr_idx], y_float[tr_idx])
                pred = pipe_fold.predict(X_labeled.iloc[va_idx])
                oof[va_idx] = pred
                rmse = np.sqrt(mean_squared_error(y_float[va_idx], pred))
                r2 = r2_score(y_float[va_idx], pred)
                print(f"  Fold {fold}: RMSE={rmse:.4f}, R2={r2:.4f}")
            rmse = np.sqrt(mean_squared_error(y_float, oof))
            r2 = r2_score(y_float, oof)
            print(f"OOF: RMSE={rmse:.4f}, R2={r2:.4f}")

        # Fit on all labeled groups
        if args.model_type == 'logistic':
            pipe.fit(X_labeled, y_bin)
        else:
            pipe.fit(X_labeled, y_float)

        # Predict for all groups
        scores_group = pipe.predict_proba(df_group[feature_cols])[:, 1] if args.model_type == 'logistic' else 1 / (1 + np.exp(-pipe.predict(df_group[feature_cols])))
        df_group['extremism_score'] = scores_group.astype(float)
        # Choose threshold based on labeled groups only
        thr = choose_threshold(df_group.loc[labeled_mask_group, 'extremism_score'].to_numpy(), y_bin, args.match_prevalence, args.threshold) if args.model_type == 'logistic' else args.threshold
        print(f"Threshold for binary decision (group-level): {thr:.4f} (match_prevalence={args.match_prevalence})")
        df_group['extremism_pred'] = (df_group['extremism_score'] >= thr).astype(int)

        # Map back to original df
        df = df.merge(df_group[[id_col, 'extremism_score', 'extremism_pred']], on=id_col, how='left', suffixes=('', '_by_id'))
        labeled_mask_row = df[args.extremism_column].notna()
        unlabeled_mask_row = ~labeled_mask_row
        filled = df[args.extremism_column].copy()
        # use values to satisfy type checker
        filled.loc[unlabeled_mask_row] = df.loc[unlabeled_mask_row, 'extremism_pred'].to_numpy()
        df[args.extremism_column + '_filled'] = filled

    else:
        # No grouping, fallback to row-level training/prediction
        labeled_mask = df[args.extremism_column].notna()
        y_series = df.loc[labeled_mask, args.extremism_column].map(to_int_label)
        valid_mask = y_series.notna()
        y = y_series.loc[valid_mask].astype(int).to_numpy()
        if y.size == 0:
            raise RuntimeError("No valid labeled rows after normalization to 0/1.")
        explicit_cols = [c.strip() for c in args.primary_columns.split(',')] if args.primary_columns else None
        numeric_cols, categorical_cols = detect_feature_columns(
            df,
            args.extremism_column,
            explicit_cols,
            ignore_by_name,
            drop_categorical_when_grouping=False,
        )
        if not numeric_cols and not categorical_cols:
            raise RuntimeError("No feature columns selected. Provide --primary_columns explicitly or adjust ignore list.")
        print(f"Selected feature columns -> numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}")
        if len(numeric_cols) <= 15:
            print(f"  numeric: {numeric_cols}")
        if len(categorical_cols) <= 20:
            print(f"  categorical: {categorical_cols}")
        pipe = build_pipeline(numeric_cols, categorical_cols, args.model_type)
        X_labeled = df.loc[labeled_mask].loc[valid_mask, numeric_cols + categorical_cols]
        try:
            skf = StratifiedKFold(n_splits=min(5, max(2, np.bincount(y).min())), shuffle=True, random_state=42)
        except Exception:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_scores = np.zeros_like(y, dtype=float)
        fold = 0
        for tr_idx, va_idx in skf.split(np.zeros_like(y), y):
            fold += 1
            X_tr = X_labeled.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_labeled.iloc[va_idx]
            pipe_fold = build_pipeline(numeric_cols, categorical_cols, args.model_type)
            pipe_fold.fit(X_tr, y_tr)
            s = pipe_fold.predict_proba(X_va)[:, 1]
            oof_scores[va_idx] = s
            try:
                auc = roc_auc_score(y[va_idx], s)
                ap = average_precision_score(y[va_idx], s)
                print(f"  Fold {fold}: ROC-AUC={auc:.4f}, AP={ap:.4f}")
            except Exception:
                pass
        try:
            auc = roc_auc_score(y, oof_scores)
            ap = average_precision_score(y, oof_scores)
            print(f"OOF: ROC-AUC={auc:.4f}, AP={ap:.4f}")
        except Exception:
            pass
        pipe.fit(X_labeled, y)
        X_all = df[numeric_cols + categorical_cols]
        scores = pipe.predict_proba(X_all)[:, 1]
        df['extremism_score'] = scores.astype(float)
        unlabeled_mask = ~labeled_mask
        thr = choose_threshold(df.loc[unlabeled_mask, 'extremism_score'].to_numpy(), y, args.match_prevalence, args.threshold)
        print(f"Threshold for binary decision: {thr:.4f} (match_prevalence={args.match_prevalence})")
        df['extremism_pred'] = (df['extremism_score'] >= thr).astype(int)
        filled = df[args.extremism_column].copy()
        filled.loc[unlabeled_mask] = df.loc[unlabeled_mask, 'extremism_pred'].to_numpy()
        df[args.extremism_column + '_filled'] = filled

    if args.dry_run:
        if args.group_by_id and id_col:
            unlabeled_mask_any = df[args.extremism_column].isna()
            pos_rate_pred_unlabeled = float(df.loc[unlabeled_mask_any, 'extremism_pred'].astype(float).mean()) if unlabeled_mask_any.any() else float('nan')
            print(f"Dry-run: predicted positive rate (unlabeled rows)={pos_rate_pred_unlabeled:.3f}")
        else:
            pos_rate_labeled = float(np.mean(y)) if 'y' in locals() else float('nan')
            unlabeled_mask_any = df[args.extremism_column].isna()
            pos_rate_pred_unlabeled = float(df.loc[unlabeled_mask_any, 'extremism_pred'].astype(float).mean()) if unlabeled_mask_any.any() else float('nan')
            print(f"Dry-run: labeled positive rate={pos_rate_labeled:.3f}, predicted positive rate (unlabeled)={pos_rate_pred_unlabeled:.3f}")
        print("No files written due to --dry_run.")
        return

    # Decide output path
    if args.inplace:
        out_path = csv_path
    else:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = csv_path.with_name(csv_path.stem + '_extremism_filled' + csv_path.suffix)

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Wrote predictions and filled labels to: {out_path}")


if __name__ == '__main__':
    main()
