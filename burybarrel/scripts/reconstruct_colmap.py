import json
import os
from pathlib import Path
import shutil
import subprocess

import click
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import quaternion
import sqlite3
import trimesh
import visu3d as v3d
import yaml

from burybarrel import get_logger, add_file_handler, log_dir
import burybarrel.colmap_util as cutil
from burybarrel.image import imgs_from_dir
from burybarrel.camera import save_v3dcams, RadialCamera
from burybarrel.mesh import load_mesh


logger = get_logger(__name__)
add_file_handler(logger, log_dir / "colmap_reconstruct.log")
DEFAULT_DATA_DIR_LOCAL = Path("data/input_data/")
DEFAULT_RESULTS_DIR_LOCAL = Path("results/")


@click.command()
@click.option(
    "-n",
    "--name",
    "dataset_names",
    required=True,
    type=click.STRING,
    help="Names of all datasets to process in data_dir. Use 'all' to run all valid datasets in data_dir",
    multiple=True,
)
@click.option(
    "-d",
    "--datadir",
    "data_dir",
    default=DEFAULT_DATA_DIR_LOCAL,
    required=True,
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
    help="Directory containing all datasets",
)
@click.option(
    "-o",
    "--outdir",
    "out_dir",
    default=DEFAULT_RESULTS_DIR_LOCAL,
    required=True,
    type=click.Path(file_okay=False),
    show_default=True,
    help="Output directory for all results",
)
@click.option(
    "--sparse",
    "sparse",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run sparse reconstruction",
)
@click.option(
    "--dense",
    "dense",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run dense + mesh + texture reconstruction (requires sparse to be run first)",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite existing reconstructions if they exist",
)
@click.option(
    "--num-retries",
    "num_retries",
    default=3,
    type=click.INT,
    help="Max number of times to retry COLMAP reconstruction on failure",
)
def reconstruct_colmap(dataset_names, data_dir, out_dir, sparse, dense, overwrite, num_retries):
    """
    3D reconstruction with COLMAP + OpenMVS via brute force.

    This will retry up to a number of times if the reconstruction fails catastrophically.
    """
    if "all" in [n.lower() for n in dataset_names]:
        dataset_names = []
        alldatapaths = Path(data_dir).glob("*")
        for datapath in alldatapaths:
            if datapath.is_dir() and (datapath / "info.json").exists():
                dsname = datapath.name
                dataset_names.append(dsname)
    failures = []
    for dsname in dataset_names:
        indir = Path(data_dir) / dsname
        outdir = Path(out_dir) / dsname
        success = False
        # flag if colmap succeeded but openmvs failed
        sparse_success_dense_fail = False
        logger.info(f"BEGINNING RECONSTRUCTION for dataset {dsname}")
        nretries_data = num_retries
        i = 0
        while i < nretries_data:
            # COLMAP has a tendency of segfaulting randomly, there is no feasible way to
            # catch and prevent this
            sparse_iter = not sparse_success_dense_fail
            try:
                if i == 0:
                    _reconstruct_colmap(indir, outdir, sparse=sparse_iter, dense=dense, overwrite=overwrite)
                else:
                    _reconstruct_colmap(indir, outdir, sparse=sparse_iter, dense=dense, overwrite=True)
                success = True
                logger.info(f"RECONSTRUCTION SUCCESS for dataset {dsname}")
                break
            except InvalidCOLMAPError as e:
                if i < num_retries - 1:
                    print(f"Failed to reconstruct dataset {dsname} due to error: {e}. Retrying.")
                else:
                    print(f"Could not create proper reconstruction in dataset {dsname}: {e}")
            except OpenMVSSigSegvError as e:
                # OpenMVS randomly segfaults, but since we run it as a subprocess, we can
                # catch it. This should work with enough tries, so this shouldn't count as
                # a retry.
                sparse_success_dense_fail = True
                i -= 1
            except Exception as e:
                print(f"Failed to reconstruct dataset {dsname} due to error: {e}")
                print("This is probably some random memory error that happens uncontrollably, retry.")
            i += 1
        if not success:
            logger.error(f"RECONSTRUCTION FAILURE to reconstruct dataset {dsname} after {num_retries} attempts")
            failures.append(dsname)
    logger.info(f"FAILED RECONSTRUCTION DATASETS: {failures}")


def _reconstruct_colmap(data_dir, out_dir, f_prior=None, c_prior=None, sparse=True, dense=True, overwrite=False):
    ### colmap code ###
    data_dir = Path(data_dir)
    img_dir = data_dir / "rgb"
    out_dir = Path(out_dir)
    colmap_out = out_dir / "colmap-out"
    openmvs_out = out_dir / "openmvs-out"
    colmap_out.mkdir(parents=True, exist_ok=True)
    openmvs_out.mkdir(parents=True, exist_ok=True)
    database_path = colmap_out / "database.db"
    camposes_path = colmap_out / "cam_poses.json"
    sparseply_path = colmap_out / "sparse.ply"
    # to generate from colmap
    # treat as ground truth since we don't have any intrinsics
    camintrinsics_path = data_dir / "camera.json"
    mvs_dir = colmap_out / "mvs"
    # temporary directory for models with non-fixed focal length
    # we want to fix focal length in the actual reconstruction
    sparsetmp_dir = colmap_out / "sparse_models_tmp"
    sparsetmp_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and (openmvs_out / "scene_dense_mesh_refine_texture.obj").exists():
        logger.info(f"Output directory {out_dir} already has textured mesh, skipping")
        return

    if sparse:
        if overwrite and database_path.exists():
            database_path.unlink()
        imgpaths, imgs = imgs_from_dir(img_dir)
        # assume same size for all images (surely colmap will error if not)
        w, h = imgs[0].size
        if f_prior is None:
            # i forgot the usual prior estimate used for focal length. it was a function
            # of the image dimensions somehow though
            f_prior = 1300
        if c_prior is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = c_prior
        camera = pycolmap.Camera(
            model=pycolmap.CameraModelId.RADIAL,
            width=w,
            height=h,
            # f, cx, cy, k1, k2
            params=[f_prior, cx, cy, 0.0, 0.0]
        )
        pycolmap.extract_features(
            database_path,
            img_dir,
            camera_model=camera.model.name,
            reader_options={"camera_model": camera.model.name, "camera_params": camera.params_to_string()},
            sift_options={
                # DSP-SIFT is presumably better
                "domain_size_pooling": True,
                "edge_threshold":  5.0,
                "peak_threshold":  1 / 250,
                "max_num_orientations": 3,
                "num_octaves": 8,
                "octave_resolution": 6,
                "num_threads": 4,
                "estimate_affine_shape": True,
                "dsp_max_scale": 6.0,
                "dsp_min_scale": 0.08,
                "dsp_num_scales": 20,
            }
        )
        pycolmap.match_exhaustive(
            database_path,
            verification_options={
                "detect_watermark": False,
                "ransac": {
                    "confidence": 0.999,
                }
            }
        )
        incrementalmapping_options = {
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": True,
            "init_num_trials": 400,
            "ba_global_max_num_iterations": 100,
            "ba_global_function_tolerance": 1e-2,
            "ba_local_function_tolerance": 1e-2,
            # just double all the thresholds lol
            # surely this won't go horribly
            "mapper": {
                "abs_pose_max_error": 12.0,
                "abs_pose_min_inlier_ratio": 0.1,
                "abs_pose_min_num_inliers": 10,
                "filter_max_reproj_error": 8.0,
                "filter_min_tri_angle": 3.0,
                "init_max_error": 8.0,
                "init_max_reg_trials": 4,
                "init_min_num_inliers": 20,
                "max_reg_trials": 6,
            }
        }
        maps = pycolmap.incremental_mapping(
            database_path, img_dir, sparsetmp_dir,
            options=incrementalmapping_options,
        )
        print(f"All reconstructed without prior intrinsics maps: {maps}")
        check_maps_valid(maps)
        
        # rerun with fixed intrinsics estimated from first run (is this valid?)
        reconstruction = pycolmap.Reconstruction(sparsetmp_dir / "0")
        cams, camnames = cutil.get_cams_v3d(reconstruction, return_names=True)
        specs: RadialCamera = cams.spec
        # medians due to outlier cameras appearing sometimes
        med_f = np.median(specs.K[:, 0, 0])
        med_k1 = np.median(specs.k1k2[:, 0])
        med_k2 = np.median(specs.k1k2[:, 1])
        with pycolmap.Database(database_path) as db:
            ncams = db.num_cameras
            for i in range(ncams):
                fixedcam = pycolmap.Camera(
                    camera_id=i + 1,
                    has_prior_focal_length=True,
                    model=pycolmap.CameraModelId.RADIAL,
                    width=w,
                    height=h,
                    # f, cx, cy, k1, k2
                    params=[med_f, cx, cy, med_k1, med_k2]
                )
                db.update_camera(fixedcam)
        incrementalmapping_options["ba_refine_focal_length"] = False
        incrementalmapping_options["ba_refine_principal_point"] = False
        incrementalmapping_options["ba_refine_extra_params"] = False
        maps = pycolmap.incremental_mapping(
            database_path, img_dir, colmap_out,
            options=incrementalmapping_options
        )
        print(f"All reconstructed maps using fixed estimated intrinsics: {maps}")
        check_maps_valid(maps)
        reconstruction = pycolmap.Reconstruction(colmap_out / "0")
        cams, camnames = cutil.get_cams_v3d(reconstruction, return_names=True)
        # saving relevant information from the reconstruction
        save_v3dcams(cams, [img_dir / name for name in camnames], camposes_path)
        camintrinsics = {
            "fx": med_f.item(),
            "fy": med_f.item(),
            "cx": cx,
            "cy": cy,
            "k1": med_k1.item(),
            "k2": med_k2.item(),
            "width": w,
            "height": h,
        }
        with open(camintrinsics_path, "wt") as f:
            json.dump(camintrinsics, f, indent=4)
        pts, cols = cutil.get_pc(reconstruction)
        trimeshpc = trimesh.points.PointCloud(pts, colors=cols)
        trimeshpc.export(sparseply_path)

        if overwrite and mvs_dir.exists():
            # rmtree is probably stupid, but mvs_dir is at least guaranteed to only to be
            # a dir called mvs, and not root lol
            shutil.rmtree(mvs_dir)
        # not sure the pattern colmap stores sparse maps, I presume "0" is the best
        pycolmap.undistort_images(
            mvs_dir,
            colmap_out / "0",
            img_dir,
            output_type="COLMAP",
        )

    if dense:
        try:
            # dense reconstruction
            ### openmvs code ###
            # holy crap openmvs generates so many log files
            for logpath in openmvs_out.glob("*.log"):
                logpath.unlink()
            mvs_rel = os.path.relpath(mvs_dir, openmvs_out)  # openmvs converts colmap stuff to relative paths
            print(mvs_rel)
            subprocess.run(["InterfaceCOLMAP", "-i", mvs_rel, "-o", "scene.mvs"], cwd=openmvs_out, check=True)
            subprocess.run(["DensifyPointCloud", "scene.mvs"], cwd=openmvs_out, check=True)
            # re-export dense point cloud since openmvs exports it in a format trimesh can't read
            # openmvs exports ply in a format trimesh can't read
            # trimesh exports ply in a format openmvs can't read (it segfaults ReconstructMesh)
            # WTF?????????
            load_mesh(openmvs_out / "scene_dense.ply").export(openmvs_out / "scene_dense_trimeshvalid.ply")
            # these .dmap depth maps aren't needed after densifying, and they take a lot of space
            for dmappath in openmvs_out.glob("*.dmap"):
                dmappath.unlink()
            subprocess.run(["ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply"], cwd=openmvs_out, check=True)
            subprocess.run(["RefineMesh", "scene.mvs", "-m", "scene_dense_mesh.ply", "-o", "scene_dense_mesh_refine.mvs"], cwd=openmvs_out, check=True)
            # export as obj since openmvs exports ply textures in a format blender can't read
            subprocess.run(["TextureMesh", "scene_dense.mvs", "-m", "scene_dense_mesh_refine.ply", "-o", "scene_dense_mesh_refine_texture.mvs", "--export-type", "obj"], cwd=openmvs_out, check=True)
        except Exception as e:
            raise OpenMVSSigSegvError("Segmentation fault during OpenMVS reconstruction") from e


class InvalidCOLMAPError(Exception):
    pass

class OpenMVSSigSegvError(Exception):
    pass


def check_maps_valid(maps):
    """
    Check valididty of incremantal mapping results from COLMAP.

    Raises errors if 0 reconstructions are found or if no reconstruction has >2 registered images.
    """
    if len(maps) == 0:
        raise InvalidCOLMAPError("No valid sparse reconstruction from COLMAP.")
    reconstr_numreg = [rec.num_reg_images() for rec in maps.values()]
    if max(reconstr_numreg) <= 2:
        raise InvalidCOLMAPError("No reconstruction has >2 registered images.")
