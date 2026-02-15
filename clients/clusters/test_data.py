# check_all_clusters.py
import os
import glob
import numpy as np
import hashlib
from collections import defaultdict

OUT_DIR = r'D:\DHCN\2025-2026\HK2\CongNgheMoi\project\clients\clusters'

def _hash_sample(x: np.ndarray) -> str:
    # hash theo bytes để phát hiện trùng mẫu (exact match)
    # dùng blake2b nhanh + ổn định
    b = x.tobytes()
    return hashlib.blake2b(b, digest_size=16).hexdigest()

def summarize_npz(path: str, max_hash_samples=2000):
    data = np.load(path, allow_pickle=True)
    print("=" * 90)
    print(f"[FILE] {os.path.basename(path)}")
    print("[KEYS]", list(data.files))

    def p_arr(name):
        if name not in data.files:
            return None
        arr = data[name]
        print(f"  - {name}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size and np.issubdtype(arr.dtype, np.number):
            flat = arr.reshape(-1)
            # stats trên sample nếu quá lớn
            if flat.size > 2_000_000:
                idx = np.random.default_rng(0).choice(flat.size, size=200_000, replace=False)
                flat = flat[idx]
                print("      (stats on random sample)")
            print(f"      min={float(flat.min()):.4f} max={float(flat.max()):.4f} mean={float(flat.mean()):.4f} std={float(flat.std()):.4f}")
        return arr

    X = p_arr("X")
    y = p_arr("y")
    X_test = p_arr("X_test")
    y_test = p_arr("y_test")

    if X is not None and y is not None:
        yv = np.asarray(y).reshape(-1)
        print(f"  [CHECK] len(X)={X.shape[0]} len(y)={yv.shape[0]} -> {'OK' if X.shape[0]==yv.shape[0] else 'MISMATCH'}")
        uniq, cnt = np.unique(yv, return_counts=True)
        print("  [y distribution]", {int(u): int(c) for u, c in zip(uniq, cnt)})

    if X_test is not None and y_test is not None:
        yv = np.asarray(y_test).reshape(-1)
        print(f"  [CHECK] len(X_test)={X_test.shape[0]} len(y_test)={yv.shape[0]} -> {'OK' if X_test.shape[0]==yv.shape[0] else 'MISMATCH'}")
        uniq, cnt = np.unique(yv, return_counts=True)
        print("  [y_test distribution]", {int(u): int(c) for u, c in zip(uniq, cnt)})

    # trả về hashes để kiểm overlap
    hashes = set()
    if X is not None:
        n = min(int(X.shape[0]), int(max_hash_samples))
        for i in range(n):
            hashes.add(_hash_sample(np.asarray(X[i])))
    return hashes, int(X.shape[0]) if X is not None else 0

def main():
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"OUT_DIR not found: {OUT_DIR}")

    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in: {OUT_DIR}")

    print(f"[SCAN] dir={OUT_DIR} -> {len(files)} files")
    print("\n".join(["  - " + os.path.basename(f) for f in files]))
    print()

    file_hashes = {}
    file_sizes = {}

    # 1) summarize each
    for f in files:
        hs, n = summarize_npz(f)
        file_hashes[f] = hs
        file_sizes[f] = n

    # 2) overlap check (client-client)
    print("\n" + "#" * 90)
    print("[OVERLAP CHECK] (exact duplicate samples, based on hashing X[i])")
    checked = 0
    overlaps = 0
    fs = list(file_hashes.keys())

    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            a, b = fs[i], fs[j]
            inter = file_hashes[a].intersection(file_hashes[b])
            checked += 1
            if inter:
                overlaps += 1
                print(f"  !! OVERLAP {os.path.basename(a)} <-> {os.path.basename(b)} : {len(inter)} duplicated samples (within hashed subset)")
            else:
                print(f"  OK  {os.path.basename(a)} <-> {os.path.basename(b)} : no duplicates (within hashed subset)")

    print(f"\n[SUMMARY] pairs_checked={checked}, pairs_with_overlap={overlaps}")
    print("Note: overlap check chỉ đảm bảo 'không trùng' trong subset đã hash (mặc định 2000 mẫu/file).")
    print("#" * 90)

if __name__ == "__main__":
    main()
