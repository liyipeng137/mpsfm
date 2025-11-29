import argparse
from pathlib import Path

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="local/example", help="Main data dir storing inputs and later the outputs"
)
parser.add_argument("--images_dir", type=str, help="Images directory")
parser.add_argument(
    "--imnames", type=str, nargs="*", help="List of image names to process. Leave empty to process all images"
)
parser.add_argument("--intrinsics_pth", type=str, default=None, help="Path to intrinsics .yaml file")
parser.add_argument("--refrec_dir", type=str, default=None, help="Path to reference reconstruction")
parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
parser.add_argument("-e", "--extract", nargs="*", type=str, default=[], help="List of priors to force reextract")
parser.add_argument("-c", "--conf", type=str, help="Name of the sfm config file", default="sp-lg_m3dv2")
parser.add_argument("-v", "--verbose", type=int, default=0)
parser.add_argument("--external_depth_dir", type=str, default=None, help="Path to external depth images directory (uint16 PNG in mm)")
parser.add_argument("--external_normal_dir", type=str, default=None, help="Path to external normal images directory (RGB PNG)")
parser.add_argument("--external_depth_conf_dir", type=str, default=None, help="Path to external depth confidence directory (uint8 PNG [0-255])")
parser.add_argument("--skip_masks", action="store_true", help="Skip mask extraction (e.g., sky segmentation)")

args, _ = parser.parse_known_args()
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml", return_name=False)
conf.extract = args.extract
conf.verbose = args.verbose

experiment = SimpleTest(conf)
mpsfm_rec = experiment(
    imnames=args.imnames,
    intrinsics_pth=args.intrinsics_pth,
    refrec_dir=Path(args.refrec_dir) if args.refrec_dir else None,
    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    data_dir=Path(args.data_dir),
    images_dir=Path(args.images_dir) if args.images_dir else None,
    external_depth_dir=Path(args.external_depth_dir) if args.external_depth_dir else None,
    external_normal_dir=Path(args.external_normal_dir) if args.external_normal_dir else None,
    external_depth_conf_dir=Path(args.external_depth_conf_dir) if args.external_depth_conf_dir else None,
    skip_masks=args.skip_masks,
)
sfm_outputs_dir = Path(args.data_dir) / "sfm_outputs"
sfm_outputs_dir.mkdir(exist_ok=True)
mpsfm_rec.write(sfm_outputs_dir)
