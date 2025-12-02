import os
import numpy as np
import numpy as np
import collections
import struct
from typing import NamedTuple
import PIL.Image as Image
import sys
import math
import cv2

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
class ImageCC(BaseImage):
    pass
    # def qvec2rotmat(self):
    #     return qvec2rotmat(self.qvec)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    # depth: np.array
    lidar_depth: np.array
    depth_confidence: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    principal_point_ndc: np.array

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ImageCC(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depth_acc=1000.0):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=='OPENCV':
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        principal_point_ndc = np.array([cx / width, cy / height])
        print(f"##########os.path.basename(extr.name): {os.path.basename(extr.name)}")

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        print(f"##########image_path: {image_path}")
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]

        base_path = os.path.dirname(images_folder)

        lidar_depth_path_npy = os.path.join(base_path, "depths", f"{image_name}.npy")
        if os.path.exists(lidar_depth_path_npy):
            lidar_depth = np.load(lidar_depth_path_npy).astype(np.float32)
            if lidar_depth.shape[0] != img_h or lidar_depth.shape[1] != img_w:
                # resize depth to match image resolution
                lidar_depth = cv2.resize(lidar_depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            raise FileNotFoundError(f"Lidar depth file not found: {lidar_depth_path_npy}")

        # load confidence
        confidence_path = os.path.join(base_path, "confidence", f"{image_name}.png")
        if os.path.exists(confidence_path):
            conf_read = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)
            if conf_read is not None:
                confidence = conf_read[..., 0].astype(np.float32) / 255.0
                if confidence.shape[0] != img_h or confidence.shape[1] != img_w:
                    confidence = cv2.resize(confidence, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                confidence = None
        else:
            confidence = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              lidar_depth=lidar_depth, 
                              depth_confidence=confidence,
                            #   depth=depth,
                              principal_point_ndc=principal_point_ndc)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


import open3d as o3d
import random
def get_confidence_path(image_path):
    image_filename = os.path.basename(image_path)
    parent_dir = os.path.dirname(os.path.dirname(image_path))
    confidence_path = os.path.join(parent_dir, "confidence", image_filename.replace(".jpg", ".png"))
    return confidence_path

def generate_ply_from_rgbd(train_cam_infos, meta, num_points, ply_path, cam_intrinsics=None):
    print("Generating ply from rgbd")
    train_example = train_cam_infos[0]
    w, h = train_example.width, train_example.height

    samples_per_frame = (num_points + len(train_cam_infos)) // len(train_cam_infos)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.1,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    points_list = []
    colors_list = []
    if cam_intrinsics is not None:
        fx, fy, cx, cy = cam_intrinsics[1].params[0], cam_intrinsics[1].params[1], cam_intrinsics[1].params[2], cam_intrinsics[1].params[3]
    else:
        if "fl_x" in meta:
            fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
        else:
            try:
                fx, fy, cx, cy = meta["frames"][0]["fl_x"], meta["frames"][0]["fl_y"], meta["frames"][0]["cx"], meta["frames"][0]["cy"]
            except KeyError:
                # raise exception
                print("Error: No intrinsics found")
                quit()
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    for train_cam in train_cam_infos:
        w2c = getWorld2View2(train_cam.R, train_cam.T)
        c2w = np.linalg.inv(w2c)

        image_path = train_cam.image_path
        color = cv2.imread(image_path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(color)

        # depth = (train_cam.depth * 1000).astype(np.uint16)
        # train_cam.lidar_depth_confidence
        depth = (train_cam.lidar_depth * 1000).astype(np.uint16)
        confidence_path = get_confidence_path(image_path)

        conf = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)
        # 二值化处理，阈值为200
        threshold_value = 200
        _, binary_conf = cv2.threshold(conf, threshold_value, 255, cv2.THRESH_BINARY)
        conf_mask = binary_conf // 255 
        # 应用confidence
        depth = depth * conf_mask

        depth = o3d.geometry.Image(depth)

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            np.linalg.inv(c2w),
        )
    
    # 直接提取 mesh
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(ply_path, mesh)
    #     pcd = volume.extract_point_cloud()

    #     samples_per_frame = min(samples_per_frame, len(pcd.points))
    #     mask = random.sample(range(len(pcd.points)), samples_per_frame)
    #     mask = np.asarray(mask)
    #     color = np.asarray(pcd.colors)[mask]
    #     point = np.asarray(pcd.points)[mask]

    #     points_list.append(np.asarray(point))
    #     colors_list.append(np.asarray(color))

    # points = np.concatenate(points_list, axis=0)
    # colors = np.concatenate(colors_list, axis=0)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.io.write_point_cloud(ply_path, pcd)




data_folder = "sfm_outputs"
cameras_extrinsic_file = os.path.join(data_folder, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(data_folder, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


print(f"cam_intrinsics: {cam_intrinsics}")
reading_dir = "image"
cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(data_folder, reading_dir), 
                                           depth_acc=1000)
cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


train_cam_infos = cam_infos


num_pts = 1000_000
ply_path = os.path.join(data_folder, "tsdf_result.ply")
generate_ply_from_rgbd(train_cam_infos, meta=None, num_points=num_pts, ply_path=ply_path, cam_intrinsics=cam_intrinsics)