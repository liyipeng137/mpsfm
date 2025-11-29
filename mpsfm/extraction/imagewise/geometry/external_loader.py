"""External depth and normal data loader for MP-SfM pipeline."""

from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def load_external_depth(image_name, depth_dir, conf_dir):
    """Load external depth and confidence data.
    
    Args:
        image_name: Name of the image (e.g., 'img001.png' or 'img001.npy')
        depth_dir: Directory containing float32 depth .npy files (in meters) or uint16 PNG (in millimeters)
        conf_dir: Directory containing uint8 confidence PNGs [0-255]
        
    Returns:
        dict with keys: depth, depth_variance, depth_confidence, valid
    """
    # Try .npy first, then fallback to original image extension
    image_stem = Path(image_name).stem
    depth_npy_path = Path(depth_dir) / f"{image_stem}.npy"
    depth_png_path = Path(depth_dir) / image_name
    
    # Load depth - prioritize .npy format (float32, meters)
    if depth_npy_path.exists():
        depth = np.load(str(depth_npy_path)).astype(np.float32)
        if depth is None or depth.size == 0:
            raise ValueError(f"Failed to load depth .npy: {depth_npy_path}")
    elif depth_png_path.exists():
        # Fallback: uint16 PNG (millimeters -> meters)
        depth_mm = cv2.imread(str(depth_png_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise ValueError(f"Failed to load depth PNG: {depth_png_path}")
        depth = depth_mm.astype(np.float32) / 1000.0
    else:
        raise FileNotFoundError(f"Depth file not found: {depth_npy_path} or {depth_png_path}")
    
    # Load confidence (uint8 PNG [0-255])
    conf_path = Path(conf_dir) / f"{image_stem}.png"
    conf_npy_path = Path(conf_dir) / f"{image_stem}.npy"
    
    if conf_npy_path.exists():
        # Load .npy confidence (assume already [0,1])
        confidence = np.load(str(conf_npy_path)).astype(np.float32)
    elif conf_path.exists():
        # Load PNG confidence
        conf_uint8 = cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE)
        if conf_uint8 is None:
            raise ValueError(f"Failed to load confidence image: {conf_path}")
        confidence = conf_uint8.astype(np.float32) / 255.0
    else:
        raise FileNotFoundError(f"Confidence file not found: {conf_path} or {conf_npy_path}")
    
    # Load confidence (uint8 [0-255] -> float32 [0-1])
    conf_uint8 = cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE)
    if conf_uint8 is None:
        raise ValueError(f"Failed to load confidence image: {conf_path}")
    confidence = conf_uint8.astype(np.float32) / 255.0
    
    # Calculate variance: error = depth * (1 - confidence)
    error = depth * (1 - confidence)
    depth_variance = error ** 2
    
    # Valid mask: non-zero depth
    valid = depth > 0
    
    return {
        'depth': depth,
        'depth_variance': depth_variance,
        'depth_confidence': confidence,
        'valid': valid.astype(np.float32)
    }


def load_external_normal(image_name, normal_dir, fixed_variance_degrees=5.0):
    """Load external normal data.
    
    Args:
        image_name: Name of the image (e.g., 'img001.png')
        normal_dir: Directory containing RGB normal PNGs
        fixed_variance_degrees: Fixed angle variance in degrees (default: 5 degrees)
        
    Returns:
        dict with keys: normals, normals_variance
    """
    normal_path = Path(normal_dir) / image_name
    
    if not normal_path.exists():
        raise FileNotFoundError(f"Normal file not found: {normal_path}")
    
    # Load normal image (BGR [0-255] -> RGB [-1,1])
    normal_bgr = cv2.imread(str(normal_path))
    if normal_bgr is None:
        raise ValueError(f"Failed to load normal image: {normal_path}")
    
    # Convert BGR to RGB and normalize to [-1, 1]
    normal_rgb = normal_bgr[:, :, ::-1]
    normals = (normal_rgb.astype(np.float32) / 127.5) - 1.0
    
    # Normalize to unit vectors
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6)
    
    # Generate fixed variance (scalar variance for angular uncertainty)
    H, W = normals.shape[:2]
    fixed_angle_std = np.deg2rad(fixed_variance_degrees)
    normals_variance = np.ones((H, W), dtype=np.float32) * (fixed_angle_std ** 2)
    
    return {
        'normals': normals,
        'normals_variance': normals_variance
    }


def convert_to_h5(image_list, external_dirs, output_h5_path, verbose=0):
    """Convert external depth and normal data to H5 format.
    
    Args:
        image_list: List of image names to process
        external_dirs: dict with keys 'depth', 'normal', 'depth_conf'
        output_h5_path: Path to output H5 file
        verbose: Verbosity level
        
    Returns:
        Path to the created H5 file
    """
    depth_dir = external_dirs.get('depth')
    normal_dir = external_dirs.get('normal')
    conf_dir = external_dirs.get('depth_conf')
    
    if verbose > 0:
        print(f"Converting {len(image_list)} external depth/normal files to H5 format...")
        print(f"  Depth dir: {depth_dir}")
        print(f"  Normal dir: {normal_dir}")
        print(f"  Confidence dir: {conf_dir}")
    
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file if present
    if output_h5_path.exists():
        output_h5_path.unlink()
    
    processed_count = 0
    skipped_count = 0
    
    for image_name in tqdm(image_list, desc="Converting external data"):
        try:
            # Load depth data
            if depth_dir is not None and conf_dir is not None:
                depth_data = load_external_depth(image_name, depth_dir, conf_dir)
            else:
                depth_data = {}
            
            # Load normal data
            if normal_dir is not None:
                normal_data = load_external_normal(image_name, normal_dir)
            else:
                normal_data = {}
            
            # Combine all data
            combined_data = {**depth_data, **normal_data}
            
            if len(combined_data) == 0:
                if verbose > 0:
                    print(f"Warning: No data found for {image_name}")
                skipped_count += 1
                continue
            
            # Write to H5
            with h5py.File(str(output_h5_path), "a", libver="latest") as fd:
                # Use image name without path as key (consistent with model output)
                key = Path(image_name).name
                if key in fd:
                    del fd[key]
                grp = fd.create_group(key)
                for k, v in combined_data.items():
                    grp.create_dataset(k, data=v)
            
            processed_count += 1
            
        except Exception as e:
            if verbose > 0:
                print(f"Warning: Failed to process {image_name}: {e}")
            skipped_count += 1
            continue
    
    if verbose > 0:
        print(f"Converted {processed_count} images, skipped {skipped_count}")
    
    return output_h5_path

