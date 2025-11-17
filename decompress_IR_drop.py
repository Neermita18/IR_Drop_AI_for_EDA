import os
import time
import shutil
import tarfile
import gzip

t = time.time()

# Script is in same folder as split files
features_path = '.'
decompress_path = './IR_drop_features_decompressed'

# 1. Create output folder
print("Creating decompress dir...")
if os.path.exists(decompress_path):
    shutil.rmtree(decompress_path)
os.makedirs(decompress_path, exist_ok=True)

# 2. Combine split tar.gz parts
print("Combining split files into power_t.tar.gz...")

output_tar = "power_t.tar.gz"

parts = sorted([f for f in os.listdir(features_path)
                if "power_t" in f and ".tar.gz" in f and len(f.split(".")) > 2])

if not parts:
    print(" No split parts found! Check filenames.")
    exit()

with open(output_tar, "wb") as outfile:
    for p in parts:
        print("Adding part:", p)
        with open(p, "rb") as infile:
            shutil.copyfileobj(infile, outfile)

# 3. Decompress all .gz inside current folder
print("\nProcessing .gz files...")

for root, dirs, files in os.walk(features_path):
    for filename in files:
        if filename.endswith(".gz"):
            filepath = os.path.join(root, filename)
            print("Decompressing:", filepath)

            out_path = filepath[:-3]  # remove .gz
            with gzip.open(filepath, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Extract tar files
            if tarfile.is_tarfile(out_path):
                print("Extracting:", out_path)
                extract_to = decompress_path
                os.makedirs(extract_to, exist_ok=True)

                with tarfile.open(out_path) as tar:
                    tar.extractall(path=extract_to)

            os.remove(out_path)

print("\nDecompress finished in %.2fs" % (time.time() - t))

