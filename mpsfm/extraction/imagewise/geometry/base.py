import pprint

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from mpsfm.extraction import load_model
from mpsfm.utils.io import list_h5_names


def extract(data, model):
    input_data = {}
    name = data["meta"]["image_name"]
    assert len(name) == 1
    name = name[0]

    print(f"[DEBUG base.extract] Processing image: {name}")
    print(f"[DEBUG base.extract] Original data['image'] shape: {data['image'].shape}")
    
    scale = model.conf.scale if hasattr(model.conf, "scale") else 1
    print(f"[DEBUG base.extract] scale: {scale}")
    
    image = (data["image"].numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    print(f"[DEBUG base.extract] After numpy+transpose, image shape: {image.shape}")
    
    if scale != 1:
        print(f"DEBUG: Resizing image from {image.shape} to {model.conf.scale}")
        image = cv2.resize(
            image,
            None,
            fx=model.conf.scale,
            fy=model.conf.scale,
            interpolation=cv2.INTER_AREA,
        )
        print(f"[DEBUG base.extract] After resize, image shape: {image.shape}")
    
    input_data["image"] = image
    input_data["meta"] = data["meta"]
    if "intrinsics" in data:
        input_data["intrinsics"] = data["intrinsics"].numpy()[0] * scale

    pred = model(input_data)
    
    # DEBUG: 打印模型返回的键和形状
    print(f"[DEBUG base.extract] pred keys after model: {list(pred.keys())}")
    if 'depth' in pred:
        print(f"[DEBUG base.extract] pred['depth'] shape: {pred['depth'].shape}")
    if 'depth_confidence' in pred:
        print(f"[DEBUG base.extract] pred['depth_confidence'] shape: {pred['depth_confidence'].shape}")
    
    pred["name"] = name
    return pred


def write(pred, output_path):
    name = pred.pop("name")
    
    # DEBUG: 打印 write 函数收到的数据
    print(f"[DEBUG base.write] Writing data for: {name}")
    print(f"[DEBUG base.write] pred keys: {list(pred.keys())}")
    if 'depth' in pred:
        print(f"[DEBUG base.write] depth shape: {pred['depth'].shape}")
    if 'depth_confidence' in pred:
        print(f"[DEBUG base.write] depth_confidence shape: {pred['depth_confidence'].shape}")
    
    with h5py.File(str(output_path), "a", libver="latest") as fd:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
    
    from pathlib import Path
    # === 新增：保存为图像文件 ===
    output_dir = Path("/kiri/tmp/")
    
    # 保存深度图
    if 'depth' in pred:
        depth_dir = output_dir / "depth_images"
        depth_dir.mkdir(exist_ok=True)
        depth = pred['depth']
        
        # 保存深度值为 uint16 PNG (毫米精度)
        depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{Path(name).stem}.png"), depth_mm)
        
        # 保存深度原始 confidence (Metric3Dv2 模型输出)
        if 'depth_confidence' in pred:
            depth_conf_dir = output_dir / "depth_confidence"
            depth_conf_dir.mkdir(exist_ok=True)
            
            confidence = pred['depth_confidence']  # 范围 [0, 1]
            # 直接映射到 [0, 255]
            confidence_vis = (confidence * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(depth_conf_dir / f"{Path(name).stem}.png"), confidence_vis)
        
        # 保存深度 variance (派生的不确定性)
        if 'depth_variance' in pred:
            depth_var_dir = output_dir / "depth_variance"
            depth_var_dir.mkdir(exist_ok=True)
            
            variance = pred['depth_variance']
            # 方差转置信度可视化
            std = np.sqrt(variance)
            confidence_from_var = np.exp(-std * 3)
            var_vis = (confidence_from_var * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(depth_var_dir / f"{Path(name).stem}.png"), var_vis)
    
    # 保存法线图
    if 'normals' in pred:
        normals_dir = output_dir / "normals_images"
        normals_dir.mkdir(exist_ok=True)
        normals = pred['normals']
        
        # 可视化保存 (RGB)
        normals_vis = ((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)
        # BGR for OpenCV
        normals_vis = normals_vis[..., ::-1]
        cv2.imwrite(str(normals_dir / f"{Path(name).stem}.png"), normals_vis)
        
        # 保存法线 confidence
        if 'normals_variance' in pred:
            normals_conf_dir = output_dir / "normals_confidence"
            normals_conf_dir.mkdir(exist_ok=True)
            
            variance = pred['normals_variance']
            # 方差转置信度
            confidence = np.exp(-variance * 10)  # 调整系数使得合理方差映射到可见范围
            confidence_vis = (confidence * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(normals_conf_dir / f"{Path(name).stem}.png"), confidence_vis)
        


@torch.no_grad()
def main(conf, export_dir, overwrite=False, image_list=None, model=None, scene_parser=None, verbose=0):
    if verbose > 0:
        print("Extracting geometry with configuration:" f"\n{pprint.pformat(conf)}")
    from pathlib import Path
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    write_name = conf.model.write_name if "write_name" in conf.model else conf.model.name
    output_path = export_dir / f"{write_name}.h5"
    skip_names = set(list_h5_names(output_path) if output_path.exists() and not overwrite else ())
    extract_num = len(image_list)
    image_list = [f for f in image_list if f not in skip_names]
    if verbose > 0:
        print(f"Skipping {extract_num-len(image_list)} files")

    loader = scene_parser.dataset(
        conf.dataset, image_list=image_list, scene_parser=scene_parser, cache_dir=export_dir
    ).get_dataloader()

    if verbose > 0:
        print(f"Extracting {len(loader)} files")
    if len(loader) == 0:
        print("Skipping the extraction.")
        return output_path, model

    if model is None:
        model = load_model(conf)

    for data in tqdm(loader):
        pred = extract(data, model)
        if pred is None:
            continue
        write(pred, output_path)

    return output_path, model
