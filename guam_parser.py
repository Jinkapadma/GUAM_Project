import os
import h5py
import numpy as np
from math import comb
from typing import Tuple, Dict

def interpHermBern(wpts: np.ndarray, tint: Tuple[float, float]) -> np.ndarray:
    """Computes Bernstein polynomial control points for Hermite interpolation."""
    o_continuity = wpts.shape[1]
    n_der = np.arange(o_continuity)
    denom = np.cumprod(np.arange(2 * o_continuity - 1, o_continuity - 1, -1))
    S = np.ones(o_continuity)
    if o_continuity > 1:
        S[1:] = 1.0 / denom[:o_continuity - 1]
    d0 = max(tint[1] - tint[0], 1e-6)
    Sd = d0 ** n_der
    ins = S * Sd * wpts[0, :]
    ins2 = S * Sd * wpts[1, :]
    Q = np.zeros(2 * o_continuity)
    for n in range(1, o_continuity + 1):
        bn = np.array([comb(n - 1, k) for k in range(n)])
        Q[n - 1] = np.dot(bn, ins[:n])
        sn = (-1) ** np.arange(n)
        Q[-n] = np.dot(sn * bn, ins2[:n])
    return Q

def evalBernPoly(P: np.ndarray, t: np.ndarray, tint: Tuple[float, float]) -> np.ndarray:
    """Evaluates a Bernstein polynomial at times t."""
    d0 = max(tint[1] - tint[0], 1e-6)
    t_norm = np.clip((t - tint[0]) / d0, 0.0, 1.0)
    n = len(P)
    Nt = len(t_norm)
    h = np.ones(Nt)
    u = 1.0 - t_norm
    Q = P[0] * h
    for k in range(1, n):
        h = h * t_norm * (n - k)
        h = h / (k * u + h + 1e-12)
        h1 = 1.0 - h
        Q = h1 * Q + h * P[k]
    return Q

def evalPWCurve(wpts: np.ndarray, times_seg: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Evaluates piecewise Bezier curve across time intervals."""
    N_wpts = len(times_seg)
    val = np.zeros_like(t_eval)
    for i in range(N_wpts - 1):
        tint = (times_seg[i], times_seg[i+1])
        if i == N_wpts - 2:
            mask = (t_eval >= tint[0]) & (t_eval <= tint[1] + 1e-5)
        else:
            mask = (t_eval >= tint[0]) & (t_eval < tint[1])
        if not np.any(mask):
            continue
        seg_wpts = np.array([wpts[i], wpts[i+1]])
        Q = interpHermBern(seg_wpts, tint)
        val[mask] = evalBernPoly(Q, t_eval[mask], tint)
    return val

def parse_guam_mat(mat_path: str, max_trajectories: int = 1000, num_pts: int = 120) -> np.ndarray:
    """
    Parses GUAM Data_Set_1.mat and extracts 3D trajectory position arrays.
    Returns array of shape (N_trajectories, num_pts, 3) in [x, y, z] (ft).
    """
    print(f"Loading GUAM dataset from: {mat_path}")
    trajectories = []
    with h5py.File(mat_path, 'r') as f:
        own_traj = f['own_traj']
        total_runs = min(own_traj.shape[1], max_trajectories)
        print(f"Extracting {total_runs} trajectories...")
        for i in range(total_runs):
            try:
                wptsX = f[own_traj[0, i]][:].T
                wptsY = f[own_traj[1, i]][:].T
                wptsZ = f[own_traj[2, i]][:].T
                tX = f[own_traj[3, i]][:].flatten()
                tY = f[own_traj[4, i]][:].flatten()
                tZ = f[own_traj[5, i]][:].flatten()

                t_total = tX[-1]
                t_eval = np.linspace(0, t_total, num_pts)
                
                x = evalPWCurve(wptsX, tX, t_eval)
                y = evalPWCurve(wptsY, tY, t_eval)
                z = evalPWCurve(wptsZ, tZ, t_eval)
                
                traj_3d = np.column_stack([x, y, z])
                trajectories.append(traj_3d)
            except Exception as e:
                print(f"Warning: skipped trajectory {i} due to error: {e}")
                continue

    trajectories = np.array(trajectories)
    print(f"Extracted {len(trajectories)} trajectories with shape {trajectories.shape}")
    return trajectories

def create_guam_train_test_split(
    mat_path: str,
    obs_len: int = 12,
    pred_len: int = 24,
    max_trajectories: int = 1000,
    test_ratio: float = 0.2
) -> Dict[str, np.ndarray]:
    """
    Splits trajectories into history (obs) and future target (pred),
    normalizes scales, and splits into train / test sets.
    """
    total_len = obs_len + pred_len
    raw_trajs = parse_guam_mat(mat_path, max_trajectories=max_trajectories, num_pts=total_len)
    
    # Calculate normalization parameters across all trajectories
    mean = np.mean(raw_trajs, axis=(0, 1), keepdims=True)
    std = np.std(raw_trajs, axis=(0, 1), keepdims=True) + 1e-6
    
    norm_trajs = (raw_trajs - mean) / std
    
    obs_trajs = norm_trajs[:, :obs_len, :]       # (N, obs_len, 3)
    pred_trajs = norm_trajs[:, obs_len:, :]      # (N, pred_len, 3)
    
    num_test = int(len(norm_trajs) * test_ratio)
    num_train = len(norm_trajs) - num_test
    
    dataset_dict = {
        'train_obs': obs_trajs[:num_train],
        'train_pred': pred_trajs[:num_train],
        'train_full': norm_trajs[:num_train],
        'test_obs': obs_trajs[num_train:],
        'test_pred': pred_trajs[num_train:],
        'test_full': norm_trajs[num_train:],
        'mean': mean.squeeze(),
        'std': std.squeeze(),
        'raw_test_full': raw_trajs[num_train:]
    }
    
    print(f"Dataset split complete: {num_train} train samples, {num_test} test samples.")
    return dataset_dict

if __name__ == '__main__':
    mat_file = r'GUAM_dir\Generic-Urban-Air-Mobility-GUAM-main\Challenge_Problems\Data_Set_1.mat'
    data = create_guam_train_test_split(mat_file, max_trajectories=100)
    print("Train obs shape:", data['train_obs'].shape)
    print("Test obs shape:", data['test_obs'].shape)
