from pathlib import Path
from typing import List

import cv2
import dataclass_array as dca
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KDTree
import torch
import visu3d as v3d


def cmapvals(vals, cmap="viridis", vmin=None, vmax=None):
    """Maps a list of values to corresponding RGB values in a matplotlib colormap."""
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgbvals = np.array(scalarMap.to_rgba(vals))
    rgbvals = rgbvals[:, :3]
    return rgbvals


def get_surface_line_traces(
    x, y, z, color="#101010", width=1, step=1, include_vertical=True, include_horizontal=True
) -> List[go.Scatter3d]:
    """
    Generates plotly traces for grid lines on a 3D surface, akin to to what 3D surfaces look
    like when plotted in MATLAB.
    """
    line_marker = dict(color=color, width=width)
    traces = []
    if include_horizontal:
        for xl, yl, zl in list(zip(x, y, z))[::step]:
            traces.append(go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", line=line_marker, name=""))
    if include_vertical:
        for xl, yl, zl in list(zip(x.T, y.T, z.T))[::step]:
            traces.append(go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", line=line_marker, name=""))
    return traces


def get_ray_trace(
    pos, raydir, length=1, width=1, color="#101010", markersize=6, markersymbol="diamond"
) -> go.Scatter3d:
    """Generates a plotly trace for a 3D ray given position and direction."""
    line_marker = dict(color=color, width=width)
    pos = np.array(pos)
    raydir = np.array(raydir)
    raydir = raydir / np.linalg.norm(raydir)
    pts = np.array([
        pos,
        pos + raydir * length
    ])
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines+markers",
        line=line_marker,
        marker=go.scatter3d.Marker(size=[0, markersize, 0],symbol=markersymbol, opacity=1),
        name=""
    )


def segment_pc_from_mask(pc: v3d.Point3d, mask, v3dcam: v3d.Camera):
    idxs = np.arange(pc.shape[0])
    H, W = v3dcam.spec.resolution
    pxpts = v3dcam.px_from_world @ pc
    uvs = pxpts.p
    valid = (uvs[:, 0] >= 0) & (uvs[:, 0] <= W) & (uvs[:, 1] >= 0) & (uvs[:, 1] <= H)
    barrelmask = mask[uvs[valid].astype(int).T[1], uvs[valid].astype(int).T[0]] > 0
    barrelidxs = idxs[valid][barrelmask]
    return barrelidxs


def get_bbox_mask(bbox, W, H):
    """
    Sets values inside bounding box to 255.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max]
    """
    bbox = np.array(bbox, dtype=int)
    boxmask = np.zeros((H, W), dtype=np.uint8)
    boxmask = cv2.rectangle(boxmask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
    return boxmask


def get_local_plane_mask(bbox, expandratio_in, expandratio_out, W, H):
    """
    Takes the difference between a larger bbox and an even larger bbox mask to get
    a 'frame' of the local plane around the barrel.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max]
        expandratio_in: expansion ratio of bbox sides for inner bbox
        expandratio_out: expansion ratio of bbox sides for outer bbox
    """
    newbboxout = np.zeros(4, dtype=int)
    newbboxin = np.zeros(4, dtype=int)
    expandratioin = expandratio_in
    expandratioout = expandratio_out
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    cx = bbox[0] + width // 2
    cy = bbox[1] + height // 2
    newbboxin[0] = max(cx - (expandratioin * width) // 2, 0)
    newbboxin[1] = max(cy - (expandratioin * height) // 2, 0)
    newbboxin[2] = min(cx + (expandratioin * width) // 2, W)
    newbboxin[3] = min(cy + (expandratioin * height) // 2, H)
    newbboxout[0] = max(cx - (expandratioout * width) // 2, 0)
    newbboxout[1] = max(cy - (expandratioout * height) // 2, 0)
    newbboxout[2] = min(cx + (expandratioout * width) // 2, W)
    newbboxout[3] = min(cy + (expandratioout * height) // 2, H)
    return get_bbox_mask(newbboxout, W, H) - get_bbox_mask(newbboxin, W, H)


def rotate_pts_to_ax(pts, normal, target, ret_R=False):
    normal = np.array(normal, dtype=float)
    target = np.array(target, dtype=float)
    ang = np.arccos((target @ normal) / (np.linalg.norm(target) * np.linalg.norm(normal)))
    rotax = np.cross(normal, target)
    eta = (rotax / np.linalg.norm(rotax))
    theta = eta * ang
    thetahat = np.array([
        [0, -theta[2], theta[1]],
        [theta[2], 0, -theta[0]],
        [-theta[1], theta[0], 0]
    ])
    R = np.eye(3) + (np.sin(ang) / ang) * thetahat + ((1 - np.cos(ang)) / ang**2) * (thetahat @ thetahat)
    rotscenexyz = (R @ pts.T).T
    if ret_R:
        return rotscenexyz, R
    return rotscenexyz


def rotate_pts_to_ax_torch(pts, normal, target):
    """ Given a point cloud with normal vector, rotate it such that the new normal matches the target vector
    Args:
		pts: (torch.tensor) [N, 3] point cloud
		normal (torch.tensor)[3,] normal vector
		target (torch.tensor) [3,] target normal vector
    Return:
		rotated_pts (torch.tensor) [N, 3] rotated point cloud 
    """ 
    ang = torch.arccos((target @ normal)/(torch.linalg.norm(target)*torch.linalg.norm(normal)))
    rotax = torch.cross(normal, target)
    eta = (rotax / torch.linalg.norm(rotax))
    theta = eta * ang
    thetahat = torch.tensor([
        [0, -theta[2], theta[1]],
        [theta[2], 0, -theta[0]],
        [-theta[1], theta[0], 0]
    ])
    R = torch.eye(3) + (torch.sin(ang) / ang) * thetahat + ((1 - torch.cos(ang)) / ang**2) * (thetahat @ thetahat)
    rotscenexyz = pts @ R.T
    return rotscenexyz


def icp_translate(source_pc, target_pc, max_iters=20, tol=1e-3, verbose=False, ntheta=3, nphi=3):
    """
    Extremely jank implementation of iterative closest point for only translation.
    
    Initializes guesses of translation by sampling points on a sphere around the
    target point cloud.
    
    source_pc assumed to be smaller than target_pc
    
    Returns:
        translation: 3d numpy array
    """
    src_mean = np.mean(source_pc, axis=0)
    targ_mean = np.mean(target_pc, axis=0)
    scale = np.max(np.linalg.norm(target_pc - targ_mean, axis=1))
    target_kd = KDTree(target_pc)

    if ntheta > 0 and nphi > 0:
        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        phis = np.linspace(0, np.pi, nphi + 2)[1:-1]
        alltheta, allphi = np.meshgrid(thetas, phis)
        alltheta = alltheta.reshape(-1)
        allphi = allphi.reshape(-1)
        offset_choices = scale * np.array([np.sin(allphi) * np.cos(alltheta), np.sin(allphi) * np.sin(alltheta), np.cos(allphi)]).T
    else:
        offset_choices = np.array([None])
    alltranslations = np.zeros((len(offset_choices), 3))
    allmeandists = np.zeros(len(offset_choices))
    for j, offset in enumerate(offset_choices):
        # p = targ_mean - src_mean
        if offset is None:
            p = np.array([0.0, 0.0, 0.0])
        else:
            p = (targ_mean + offset) - src_mean
        prevp = p
        prevdist = np.inf
        K = max_iters
        for i in range(K):
            dists, close_idxs = target_kd.query(source_pc + p)
            meandist = np.mean(dists)
            targ_mean_filt = np.mean(target_pc[close_idxs], axis=0)
            p = targ_mean_filt - src_mean
            if np.abs(prevdist - meandist) < tol:
                if verbose:
                    print(f"converged at iter {i}")
                break
            prevp = p
            prevdist = meandist
            if i == K - 1:
                if verbose:
                    print(f"max iters {K} reached before tolerance {tol}")
        allmeandists[j] = np.mean(meandist)
        alltranslations[j, :] = p
    bestidx = np.argmin(allmeandists)
    return alltranslations[bestidx]
