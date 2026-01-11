import os
import sys
import json
import argparse
import warnings
import logging
import pickle
import math
import concurrent.futures
import hashlib
import random
from datetime import datetime

# --- warning suppression ---
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from scipy.spatial import cKDTree

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- config ---
ALLOWED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
BATCH_SIZE = 32
ATLAS_SIZE = 4096
CACHE_FILENAME = "embeddings.pkl"
HTML_TEMPLATE_FILENAME = "atlas.html"

def get_file_hash(filepath):
    """Calculates MD5 hash of a file to detect duplicates."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def get_metadata(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    tags = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tags = data.get("tags", [])
        except:
            pass
    return tags

def load_cache(project_dir):
    cache_path = os.path.join(project_dir, CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            print("  ! Cache file corrupted. Rebuilding.")
    return {}

def save_cache(project_dir, data):
    cache_path = os.path.join(project_dir, CACHE_FILENAME)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def relax_coords(coords, target_dist=1.0, iterations=50):
    """
    Physics simulation for coordinates.
    """
    N = len(coords)
    if N == 0: return coords

    print(f"Relaxing layout (Physics Simulation, {iterations} steps)...")

    # 1. Pre-Scale based on density heuristic
    min_xy = np.min(coords, axis=0)
    max_xy = np.max(coords, axis=0)
    width = max_xy[0] - min_xy[0]

    if width > 0:
        target_scale = math.sqrt(N) * 1.5
        current_scale = width
        scale_factor = target_scale / current_scale
        coords = coords * scale_factor

    # 2. Physics Loop
    for i in tqdm(range(iterations), desc="  Physics", leave=False):
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=target_dist)

        if not pairs:
            break

        moves = np.zeros_like(coords)
        counts = np.zeros(N)

        for (idx1, idx2) in pairs:
            p1 = coords[idx1]
            p2 = coords[idx2]

            delta = p1 - p2
            dist = np.linalg.norm(delta)

            if dist < 0.0001:
                delta = np.random.randn(2)
                delta /= np.linalg.norm(delta)
                dist = 0.0001

            overlap = target_dist - dist
            repulsion = (delta / dist) * (overlap * 0.5)

            moves[idx1] += repulsion
            moves[idx2] -= repulsion
            counts[idx1] += 1
            counts[idx2] += 1

        active_mask = counts > 0
        coords[active_mask] += (moves[active_mask] / counts[active_mask, None]) * 0.8

    return coords

def build_single_atlas(args):
    """Worker function for creating a single texture atlas."""
    atlas_idx, keys, current_files, thumb_dir, total_atlases, thumb_size = args
    images_per_row = ATLAS_SIZE // thumb_size

    # Pre-calculate target size tuple to avoid recreating it in the loop
    target_dims = (thumb_size, thumb_size)

    atlas_img = Image.new('RGBA', (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))
    print(f"  [Start] Atlas {atlas_idx + 1}/{total_atlases} ({len(keys)} images)...")

    results = []
    for i, key in enumerate(keys):
        full_path = current_files[key]['full_path']

        # Math for position
        col = i % images_per_row
        row = i // images_per_row
        x_pos = col * thumb_size
        y_pos = row * thumb_size

        try:
            with Image.open(full_path) as img:
                # OPTIMIZATION 1: JPEG Draft Mode
                # If it's a JPEG, load a lower-res version directly from the file header
                # instead of decoding all pixels of a 4K image just to shrink it.
                if img.format == 'JPEG':
                    img.draft('RGB', target_dims)

                # Convert to RGBA
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # OPTIMIZATION 2: BICUBIC Resampling
                # LANCZOS is too heavy for batch thumbnails. BICUBIC is 2-3x faster here.
                img.thumbnail(target_dims, Image.Resampling.BICUBIC)

                # OPTIMIZATION 3: Direct Paste (No intermediate 'thumb_square')
                # Calculate centering offsets and paste directly to atlas
                w, h = img.size
                off_x = (thumb_size - w) // 2
                off_y = (thumb_size - h) // 2

                atlas_img.paste(img, (x_pos + off_x, y_pos + off_y))

                results.append({
                    "key": key,
                    "atlas_id": atlas_idx,
                    "atlas_x": x_pos,
                    "atlas_y": y_pos
                })
        except Exception:
            # Silently skip bad image files
            pass

    save_path = os.path.join(thumb_dir, f"atlas_{atlas_idx}.webp")

    # OPTIMIZATION 4: WebP Method
    # method=6 is extremely slow (best compression).
    # method=4 provides a huge speedup with very little file size difference.
    atlas_img.save(save_path, format="WEBP", lossless=True, quality=100, method=4)

    print(f"  [Done ] Atlas {atlas_idx + 1}/{total_atlases}")
    return results

def create_atlases_parallel(sorted_keys, current_files, project_dir, thumb_size):
    print("\nGenerating Texture Atlases (Parallel)...")
    thumb_dir = os.path.join(project_dir, "thumbs")
    if not os.path.exists(thumb_dir): os.makedirs(thumb_dir)

    images_per_row = ATLAS_SIZE // thumb_size
    images_per_atlas = images_per_row * images_per_row
    total_atlases = math.ceil(len(sorted_keys) / images_per_atlas)

    tasks = []
    for atlas_idx in range(total_atlases):
        start_idx = atlas_idx * images_per_atlas
        end_idx = min(start_idx + images_per_atlas, len(sorted_keys))
        chunk_keys = sorted_keys[start_idx:end_idx]
        tasks.append((atlas_idx, chunk_keys, current_files, thumb_dir, total_atlases, thumb_size))

    all_atlas_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(build_single_atlas, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                all_atlas_data.extend(future.result())
            except Exception as e:
                print(f"  ! Worker failed: {e}")
    return all_atlas_data

def copy_and_configure_html(project_dir, thumb_size):
    """
    Reads the HTML template, replaces the thumbSize configuration, and saves it
    to the project directory as 'index.html'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, HTML_TEMPLATE_FILENAME)
    output_path = os.path.join(project_dir, "index.html")

    if not os.path.exists(template_path):
        print(f"Error: HTML template '{HTML_TEMPLATE_FILENAME}' not found at {template_path}. Skipping visualization setup.")
        print("Please ensure the HTML file is in the same directory as the Python script.")
        return

    try:
        # 1. Read the template content
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 2. Find and substitute THUMB_SIZE in the CONFIG block
        old_config_thumb_line = "thumbSize: 128,"
        new_config_thumb_line = f"thumbSize: {thumb_size},"

        # We replace the line 'thumbSize: 128,' with the dynamically sized line
        if old_config_thumb_line in html_content:
            html_content = html_content.replace(old_config_thumb_line, new_config_thumb_line)
        else:
            print("Warning: Could not find 'thumbSize: 128,' configuration line in HTML template. Visualization may use default size.")

        # 3. Write the new index.html file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Visualization HTML (index.html) created in {project_dir}")

    except Exception as e:
        print(f"Error configuring and copying HTML file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Image Embedding and Clustering Generator.")
    parser.add_argument("project_dir", help="Directory where embeddings, atlases, and index.html will be stored.")
    parser.add_argument("data_dir", help="Directory containing the source images.")
    parser.add_argument("--thumb_size", type=int, default=128, help="Resolution (in pixels) for the thumbnail size in the atlas. Must be a divisor of 4096.")
    parser.add_argument("--force", action="store_true", help="Force recalculation of all embeddings.")
    args = parser.parse_args()

    # --- Setup Variables ---
    project_dir = os.path.abspath(args.project_dir)
    data_dir = os.path.join(project_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_json = os.path.join(project_dir, "clusters.json")
    thumb_size = args.thumb_size

    if ATLAS_SIZE % thumb_size != 0:
        print(f"Error: ATLAS_SIZE ({ATLAS_SIZE}) must be divisible by THUMB_SIZE ({thumb_size}).")
        sys.exit(1)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Project:     {os.path.basename(project_dir)}")
    print(f"Input:       {data_dir}")
    print(f"Thumb Size:  {thumb_size}px")
    print(f"Hardware:    {device_type.upper()}")
    print("")

    if not os.path.exists(project_dir): os.makedirs(project_dir)

    # 1. Scan & Deduplicate
    print("Scanning & Deduplicating files...")
    raw_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTS:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                raw_files.append((rel_path, full_path, os.path.getmtime(full_path)))

    if not raw_files:
        print("Error: No images found.")
        return

    by_hash = {}
    print(f"  Hashing {len(raw_files)} files...")
    for item in tqdm(raw_files, unit="file", leave=False):
        rel_path, full_path, mtime = item
        f_hash = get_file_hash(full_path)
        if f_hash:
            if f_hash not in by_hash: by_hash[f_hash] = []
            by_hash[f_hash].append(item)

    final_files = {}
    duplicates_removed = 0

    for f_hash, items in by_hash.items():
        if len(items) > 1:
            items.sort(key=lambda x: x[2], reverse=True)
            duplicates_removed += (len(items) - 1)
        best = items[0]
        final_files[best[0]] = {'full_path': best[1], 'mtime': best[2]}

    print(f"  Total Unique: {len(final_files)}")
    print(f"  Duplicates:   {duplicates_removed}")
    print("")

    # 2. Sync Cache
    cache = {} if args.force else load_cache(project_dir)
    to_process = []

    for rel_path, info in final_files.items():
        if rel_path not in cache or cache[rel_path]['mtime'] != info['mtime']:
            to_process.append(rel_path)

    cached_paths = list(cache.keys())
    for rel_path in cached_paths:
        if rel_path not in final_files:
            del cache[rel_path]

    print(f"  To Embed: {len(to_process)}")
    print("")

    # 3. Embed
    if to_process:
        print(f"Embedding {len(to_process)} images...")
        try:
            model = SentenceTransformer('clip-ViT-B-32', device=device_type)
            pbar = tqdm(range(0, len(to_process), BATCH_SIZE), unit="batch", leave=False)
            for i in pbar:
                batch_keys = to_process[i : i + BATCH_SIZE]
                batch_imgs = []
                valid_keys = []
                for key in batch_keys:
                    try:
                        img = Image.open(final_files[key]['full_path']).convert('RGB')
                        batch_imgs.append(img)
                        valid_keys.append(key)
                    except: continue
                if not batch_imgs: continue
                batch_emb = model.encode(batch_imgs, batch_size=BATCH_SIZE, show_progress_bar=False)
                for key, emb in zip(valid_keys, batch_emb):
                    cache[key] = {'vector': emb, 'mtime': final_files[key]['mtime']}
            save_cache(project_dir, cache)
        except Exception as e:
            print(f"Embedding failed: {e}")
            return

    # 4. Map & Cluster
    sorted_keys = sorted(cache.keys())
    if not sorted_keys:
        print("No embeddings to process.")
        return

    vectors = np.array([cache[k]['vector'] for k in sorted_keys])

    print("Dimensionality Reduction (UMAP)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.01, metric='cosine', random_state=42, n_jobs=1)
    coords = reducer.fit_transform(vectors)

    print("Clustering (HDBSCAN)...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    labels = clusterer.fit_predict(coords)

    # --- RELAX ---
    coords = relax_coords(coords, target_dist=1.0, iterations=50)

    # 5. Atlases
    atlas_map_list = create_atlases_parallel(sorted_keys, final_files, project_dir, thumb_size)
    atlas_lookup = { item['key']: item for item in atlas_map_list }

    # 6. Export
    print("Exporting data...")
    output_data = []
    for i, key in enumerate(sorted_keys):
        if key not in atlas_lookup: continue
        full_path = final_files[key]['full_path']

        # Create image URL path relative to the index.html (which is in project_dir)
        # The image file is at project_dir/data_dir_name/image_rel_path
        image_url_path = os.path.join(os.path.basename(data_dir), key)
        image_url_path = image_url_path.replace("\\", "/")

        mtime = final_files[key]['mtime']
        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

        atlas_info = atlas_lookup[key]
        output_data.append({
            "image": image_url_path,
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "cluster": int(labels[i]),
            "tags": get_metadata(full_path),
            "date": date_str,
            "atlas_id": atlas_info['atlas_id'],
            "atlas_x": atlas_info['atlas_x'],
            "atlas_y": atlas_info['atlas_y']
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # 7. Configure and copy HTML
    copy_and_configure_html(project_dir, thumb_size)

    print(f"Process complete. Output: {output_json}")

if __name__ == "__main__":
    try: torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
