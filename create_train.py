import os
import random
import shutil
from pathlib import Path

# -------- CONFIG --------
DATA_ROOT = "IR_drop_features_decompressed"     
OUT_ROOT = "subset_fixed"                      
TRAIN_N = 50
TEST_N = 10

FEATURE_DIRS = ["power_i", "power_s", "power_sca", "power_all", "power_t"]
LABEL_DIR = "IR_drop"                           # label folder name

os.makedirs(OUT_ROOT, exist_ok=True)


def safe_listdir(path):
    if not path.exists():
        return []
    return sorted([f.name for f in path.iterdir() if f.is_file()])

feature_files = []
for feat in FEATURE_DIRS:
    folder = Path(DATA_ROOT) / feat
    feature_files.append(set(safe_listdir(folder)))


label_files = set(safe_listdir(Path(DATA_ROOT) / LABEL_DIR))

common_samples = set.intersection(*feature_files, label_files)

print(f"Total usable samples (all features + label exist): {len(common_samples)}")

if len(common_samples) < (TRAIN_N + TEST_N):
    raise ValueError(" Not enough complete samples! Reduce TRAIN_N / TEST_N.")

common_samples = sorted(list(common_samples))

selected = random.sample(common_samples, TRAIN_N + TEST_N)
train_samples = selected[:TRAIN_N]
test_samples = selected[TRAIN_N:]

print(f"Train samples: {len(train_samples)}")
print(f"Test samples : {len(test_samples)}")

def copy_files(sample_list, split_name):
    for feat in FEATURE_DIRS + [LABEL_DIR]:
        out_dir = Path(OUT_ROOT) / split_name / feat
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in sample_list:
            src = Path(DATA_ROOT) / feat / fname
            dst = out_dir / fname

            if src.exists():
                shutil.copy(src, dst)
            else:
                print(f"WARNING: Missing {src} (skipped)")

copy_files(train_samples, "train")
copy_files(test_samples, "test")

print("\n Subset creation complete!")
