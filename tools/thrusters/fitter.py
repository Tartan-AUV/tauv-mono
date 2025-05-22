#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def detect_deadband(df, tol_frac=0.01):
    maxT = df['T'].abs().max()
    tol = tol_frac * maxT

    # positive deadband: smallest ω>0 such that T > tol
    pos = df[df['ω'] > 0]
    mask_pos = pos['T'].abs() > tol
    ω_dead_pos = pos.loc[mask_pos, 'ω'].min() if mask_pos.any() else 0.0

    # negative deadband: largest ω<0 such that T < -tol
    neg = df[df['ω'] < 0]
    mask_neg = neg['T'].abs() > tol
    ω_dead_neg = neg.loc[mask_neg, 'ω'].max() if mask_neg.any() else 0.0

    return ω_dead_pos, ω_dead_neg

def fit_quadratic_thrust(df, mask):
    """
    Fit T = k * ω^2 on df rows where mask is True.
    Returns k.
    """
    sel = df[mask]
    ω = sel['ω'].values
    T = sel['T'].values
    X = ω**2
    k = np.dot(X, T) / np.dot(X, X)
    return k

def main():
    p = argparse.ArgumentParser(
        description="Compute thruster deadbands and T=k·ω² coefficients")
    p.add_argument('csv', help="path to CSV with columns ω (rad/s), T (N)")
    p.add_argument('--tol', type=float, default=0.01,
                   help="deadband detection tol as fraction of max |T|")
    args = p.parse_args()

    df = pd.read_csv(args.csv, header=None, names=['ω','T'])

    # 1) find deadbands
    ω_dead_pos, ω_dead_neg = detect_deadband(df, tol_frac=args.tol)

    # 2) define masks for fitting
    pos_mask = df['ω'] >= ω_dead_pos
    neg_mask = df['ω'] <= ω_dead_neg

    # 3) fit separately
    k_pos = fit_quadratic_thrust(df, pos_mask)
    k_neg = fit_quadratic_thrust(df, neg_mask)

    # 4) report
    print(f"Deadband around zero:")
    print(f"  ω ∈ ({ω_dead_neg:.2f}, {ω_dead_pos:.2f}) rad/s")
    print(f"Thrust coefficients (T = k·ω²):")
    print(f"  k_pos = {k_pos:.6e}  N/(rad/s)² (for ω ≥ {ω_dead_pos:.2f})")
    print(f"  k_neg = {k_neg:.6e}  N/(rad/s)² (for ω ≤ {ω_dead_neg:.2f})")

if __name__ == '__main__':
    main()
