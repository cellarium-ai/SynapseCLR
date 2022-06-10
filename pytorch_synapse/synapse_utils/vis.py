import torch
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from synapse_dataset import SynapseDataset
from synapse_simclr import utils
from synapse_utils import vis
from synapse_augmenter import SynapseAugmenter
from synapse_augmenter import consts as syn_consts

from scipy.ndimage import binary_erosion
import plotly.graph_objects as go
import pyvista as pv
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA as SKPCA

import warnings
from ipywidgets import interactive

from typing import Optional


# constants
color_pre = np.asarray((138,43,226)) / 255
color_cleft = np.asarray((255,128,0)) / 255
color_post = np.asarray((52,235,128)) / 255             
             

def plot_section_1_channel(
        intensity_xyz: np.ndarray,
        mask_cxyz: np.ndarray,
        section_idx: int,
        slider_axis: int,
        figsize = (3, 3),
        mask_alpha: float = 0.25,
        cleft_alpha: float = 0.9,
        show: bool = True,
        ax = None,
        fig = None,
        **imshow_kwargs):
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    section_idx = int(np.round(section_idx))
        
    if slider_axis == 0:
        img_2d = intensity_xyz[section_idx, :, :]
        mask_2d = mask_cxyz[:, section_idx, :, :]
    elif slider_axis == 1:
        img_2d = intensity_xyz[:, section_idx, :]
        mask_2d = mask_cxyz[:, :, section_idx, :]
    elif slider_axis == 2:
        img_2d = intensity_xyz[:, :, section_idx]
        mask_2d = mask_cxyz[:, :, :, section_idx]
        
    ax.imshow(
        img_2d,
        aspect=1.0,
        interpolation='nearest',
        cmap=plt.cm.Greys_r,
        **imshow_kwargs)    
    
    color_pre_alpha = np.asarray(color_pre.tolist() + [mask_alpha])
    color_cleft_alpha = np.asarray(color_cleft.tolist() + [cleft_alpha])
    color_post_alpha = np.asarray(color_post.tolist() + [mask_alpha])

    colorized_mask_2d = (
        mask_2d[0][..., None] * color_pre_alpha[None, None, :] +
        mask_2d[1][..., None] * color_cleft_alpha[None, None, :] +
        mask_2d[2][..., None] * color_post_alpha[None, None, :])
    
    ax.imshow(
        colorized_mask_2d,
        aspect=1.0,
        interpolation='nearest',
        cmap=plt.cm.Greys_r,
        **imshow_kwargs)

    ax.axis('off')
    
    if show:
        plt.show()
        
    return fig, ax


def animate_synapse(synapse_index: int):
    
    # get the intensity data
    intensity_bcxyz, mask_bcxyz = aug.augment_raw_data([synapse_dataset[synapse_index][1]])
    intensity_xyz = intensity_bcxyz[0, 0, ...].cpu().numpy()
    
    panel_size = 2
    max_z = 95

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(panel_size, panel_size))
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])    
    fig.tight_layout()

    def plot_section(z_idx, channel_idx=0):
        if z_idx > max_z:
            z_idx = 2 * max_z - z_idx
            
        ax.clear()
        ax.imshow(
            intensity_xyz[:, :, z_idx],
            cmap=plt.cm.Greys_r)
        ax.set_xticks([])
        ax.set_yticks([])

    anim = matplotlib.animation.FuncAnimation(
        fig, plot_section, frames=2 * max_z - 1, interval=50);

    plt.close()

    return anim.to_html5_video()


def plot_mask_3d(synapse_index: int, seg_mask_channels=[0, 1, 2], skip_every=5):

    # get the mask data
    intensity_bcxyz, mask_bcxyz = aug.augment_raw_data([synapse_dataset[synapse_index][1]])
    mask_cxyz = mask_bcxyz[0, :, :, :]
    final_img_size = mask_cxyz.shape[-1]
    X, Y, Z = np.mgrid[:final_img_size, :final_img_size, :final_img_size]

    data = []
    for seg_mask_channel in seg_mask_channels:
        mask_xyz = mask_cxyz[seg_mask_channel, ...].cpu().numpy()
        data.append(
            go.Scatter3d(
                    x=X[mask_xyz].flatten()[::skip_every],
                    y=Y[mask_xyz].flatten()[::skip_every],
                    z=Z[mask_xyz].flatten()[::skip_every],
                    mode='markers',
                    marker={'size': 2}))

    fig = go.Figure(data)

    fig.update_layout(
        scene = dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[0, final_img_size - 1],),
            yaxis=dict(range=[0, final_img_size - 1],),
            zaxis=dict(range=[0, final_img_size - 1],)
        ),
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))

    fig.show()
    

class SynapseVisContext:
    
    def __init__(
            self,
            dataset_path: str,
            aug_yaml_path: str,
            meta_df_override: Optional[pd.DataFrame],
            device: str = 'cpu'):
       
        # dataset
        self.synapse_dataset = SynapseDataset(dataset_path)
        if meta_df_override is not None:
            self.synapse_dataset.meta_df = meta_df_override
        
        # augmenter
        synapse_augmenter_kwargs = utils.yaml_config_hook(aug_yaml_path)
        self.aug = SynapseAugmenter(
            **synapse_augmenter_kwargs,
            device=torch.device(device),
            dtype=torch.float32)


def erode_mask(mask_xyz, rad=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sz = mask_xyz.shape[-1]
        for iz in range(sz):
            mask_xyz[:, :, iz] = binary_erosion(mask_xyz[:, :, iz], iterations=rad)
        return mask_xyz


def whiteout_missing_planes(intensity_xyz, inside_mask_xyz, threshold=0.05):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sz = intensity_xyz.shape[-1]
        for iz in range(sz):
            m = np.mean(intensity_xyz[:, :, iz].flatten()[inside_mask_xyz[:, :, iz].flatten()])
            if m < threshold:
                intensity_xyz[:, :, iz] = 1.
        return intensity_xyz
    
def get_dark_mask(intensity_xyz, inside_mask_xyz, otsu_prefactor):
    dark_mask_xyz = inside_mask_xyz.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sz = intensity_xyz.shape[-1]
        for iz in range(sz):
            try:
                intensities = intensity_xyz[:, :, iz].flatten()[inside_mask_xyz[:, :, iz].flatten()]
                t = otsu_prefactor * threshold_otsu(intensities)
                dark_mask_xyz[:, :, iz] = intensity_xyz[:, :, iz] < t
            except:
                continue
        dark_mask_xyz = dark_mask_xyz & inside_mask_xyz
        return dark_mask_xyz

def get_optimal_camera_props(mask_cxyz):

    # generate grid
    final_img_size = mask_cxyz.shape[-1]
    X, Y, Z = np.mgrid[:final_img_size, :final_img_size, :final_img_size]
    
    # mask components
    mask_pre = mask_cxyz[syn_consts.MASK_PRE_SYNAPTIC_NEURON - 1].astype(float)
    mask_post = mask_cxyz[syn_consts.MASK_POST_SYNAPTIC_NEURON - 1].astype(float)
    mask_cleft = mask_cxyz[syn_consts.MASK_SYNAPTIC_CLEFT - 1].astype(float)
    mask_all = (np.sum(mask_cxyz, 0) > 0).astype(float)
    
    # cleft COM
    cleft_com_x = np.sum(mask_cleft * X) / (1e-6 + np.sum(mask_cleft))
    cleft_com_y = np.sum(mask_cleft * Y) / (1e-6 + np.sum(mask_cleft))
    cleft_com_z = np.sum(mask_cleft * Z) / (1e-6 + np.sum(mask_cleft))
    
    # cleft normal direction
    cleft_x = X[mask_cleft > 0].flatten()
    cleft_y = Y[mask_cleft > 0].flatten()
    cleft_z = Z[mask_cleft > 0].flatten()
    cleft_points_n3 = np.vstack((cleft_x, cleft_y, cleft_z)).T
    
    try:
        cleft_pca = SKPCA(n_components=3).fit(cleft_points_n3 - np.mean(cleft_points_n3, axis=0, keepdims=True))
        cleft_normal = cleft_pca.components_[2]
        cleft_inplane_1 = cleft_pca.components_[0]
        cleft_inplane_2 = cleft_pca.components_[1]
    except:
        cleft_normal = np.asarray([0., 0., 1.])
        cleft_inplane_1 = np.asarray([1., 0., 0.])
        cleft_inplane_2 = np.asarray([0., 1., 0.])
    
    # good in-plane directions
    mask_all_x = X[mask_all > 0].flatten()
    mask_all_y = Y[mask_all > 0].flatten()
    mask_all_z = Z[mask_all > 0].flatten()
    mask_all_points_n3 = np.vstack((mask_all_x, mask_all_y, mask_all_z)).T
    mask_all_points_n3 = mask_all_points_n3 - np.mean(mask_all_points_n3, axis=0, keepdims=True)
    mask_all_points_inplane_1_n = np.einsum("ni,i->n", mask_all_points_n3, cleft_inplane_1)
    mask_all_points_inplane_2_n = np.einsum("ni,i->n", mask_all_points_n3, cleft_inplane_2)
    mask_all_points_n2 = np.vstack((mask_all_points_inplane_1_n, mask_all_points_inplane_2_n)).T
    mask_all_pca = SKPCA(n_components=2).fit(mask_all_points_n2)
    mask_all_inplane_dir_1 = mask_all_pca.components_[0]
    mask_all_inplane_dir_2 = mask_all_pca.components_[1]
    view_1 = mask_all_inplane_dir_1[0] * cleft_inplane_1 + mask_all_inplane_dir_1[1] * cleft_inplane_2 
    view_2 = mask_all_inplane_dir_2[0] * cleft_inplane_1 + mask_all_inplane_dir_2[1] * cleft_inplane_2 
    
    mask_pre_vol = 1e-6 + np.sum(mask_pre > 0)
    mask_post_vol = 1e-6 + np.sum(mask_post > 0)
    polarity = (
        (np.sum(X[mask_pre > 0]) / mask_pre_vol - np.sum(X[mask_post > 0]) / mask_post_vol) * cleft_normal[0] +
        (np.sum(Y[mask_pre > 0]) / mask_pre_vol - np.sum(Y[mask_post > 0]) / mask_post_vol) * cleft_normal[1] + 
        (np.sum(Z[mask_pre > 0]) / mask_pre_vol - np.sum(Z[mask_post > 0]) / mask_post_vol) * cleft_normal[2])
    if polarity < 0.:
        cleft_normal = (-1.) * cleft_normal

    return {
        'cleft_com_x': cleft_com_x,
        'cleft_com_y': cleft_com_y,
        'cleft_com_z': cleft_com_z,
        'cleft_normal': cleft_normal,
        'cleft_inplane_1': view_1,
        'cleft_inplane_2': view_2,
    }


def make_3d_synapse_figure(
        ctx: SynapseVisContext,
        synapse_index: int,    
        inside_erosion_radius = 3,
        mask_inflation_radis = 7,
        max_points = 200_000,
        max_triangles = 100_000,
        otsu_prefactor = 0.3,
        point_cloud_opacity = 0.1,
        point_cloud_size = 0.8,
        surface_opacity = 0.03,
        surface_point_cloud_opacity = 0.04,
        surface_point_cloud_size = 1.0,
        surface_max_points = 100_000,
        tri_alpha = 0.05,
        zoom_out = 1.75,
        fig_width = 500,
        fig_height = 500,
        view_plane = 1,
        fix_cleft_issue = True,
        show_points = False,
        fig = None,
        cam_props = None):

    # get data
    intensity_bcxyz, mask_bcxyz = ctx.aug.augment_raw_data([ctx.synapse_dataset[synapse_index][1]])
    mask_cxyz = mask_bcxyz[0, :, :, :].cpu().numpy()
    intensity_xyz = intensity_bcxyz[0, 0, ...].cpu().numpy()
    
    if fix_cleft_issue and np.sum(mask_cxyz[syn_consts.MASK_POST_SYNAPTIC_NEURON - 1]) < 1.0:
        mid = mask_cxyz.shape[-1] // 2
        mask_cxyz[syn_consts.MASK_SYNAPTIC_CLEFT - 1, mid, mid, mid] = 1

    # camera props
    if not cam_props:
        cam_props = get_optimal_camera_props(mask_cxyz)

    # generate grid
    final_img_size = mask_cxyz.shape[-1]
    X, Y, Z = np.mgrid[:final_img_size, :final_img_size, :final_img_size]
    X = (X - cam_props['cleft_com_x']) / final_img_size
    Y = (Y - cam_props['cleft_com_y']) / final_img_size
    Z = (Z - cam_props['cleft_com_z']) / final_img_size

    # preprocess
    inside_mask_xyz = np.sum(mask_cxyz, 0) > 0
    inside_mask_xyz = erode_mask(inside_mask_xyz, inside_erosion_radius)
    dark_mask_xyz = get_dark_mask(intensity_xyz, inside_mask_xyz, otsu_prefactor)
    plot_mask_xyz = dark_mask_xyz

    indices = np.random.permutation(X[plot_mask_xyz].flatten().shape[0])[:max_points]
    x = X[plot_mask_xyz].flatten()[indices]
    y = Y[plot_mask_xyz].flatten()[indices]
    z = Z[plot_mask_xyz].flatten()[indices]
    c = intensity_xyz[plot_mask_xyz].flatten()[indices]
    color = plt.cm.Greys_r(c)

    
    # make the figure from scratch
    if fig is None:
        
        data = []

        data.append(
            go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker={
                        'size': point_cloud_size,
                        'color': color,
                        'opacity': point_cloud_opacity
                    }))

        mask_bcxyz[0, 0] = ctx.aug.inflate_binary_mask(mask_bcxyz[:, 0], radius=mask_inflation_radis)[0]
        mask_bcxyz[0, 2] = ctx.aug.inflate_binary_mask(mask_bcxyz[:, 2], radius=mask_inflation_radis)[0]
        mask_cxyz = mask_bcxyz[0, :, :, :].cpu().numpy()

        for mask_int, color, plot_type in zip(
            [0, 1, 2],
            ['rgb(138,43,226)',
             'rgb(255,128,0)',
             'rgb(52,235,128)'],
            ['surface',
             'points',
             'surface']):

            if plot_type == 'surface':
                
                if show_points:
                    pre_mask_xyz = mask_cxyz[mask_int]
                    x = X[pre_mask_xyz].flatten()
                    y = Y[pre_mask_xyz].flatten()
                    z = Z[pre_mask_xyz].flatten()
                    indices = np.random.permutation(len(x))[:surface_max_points]
                    x = x[indices]
                    y = y[indices]
                    z = z[indices]

                    data.append(
                        go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                mode='markers',
                                marker={
                                    'size': surface_point_cloud_size,
                                    'color': color,
                                    'opacity': surface_point_cloud_opacity
                                }))

                pre_mask_xyz = mask_cxyz[mask_int] ^ erode_mask(mask_cxyz[mask_int].copy(), 1)
                x = X[pre_mask_xyz].flatten()
                y = Y[pre_mask_xyz].flatten()
                z = Z[pre_mask_xyz].flatten()
                indices = np.random.permutation(len(x))[:surface_max_points]
                x = x[indices]
                y = y[indices]
                z = z[indices]
                
                data.append(
                    go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode='markers',
                            marker={
                                'size': surface_point_cloud_size,
                                'color': color,
                                'opacity': surface_point_cloud_opacity
                            }))

                if show_points:
                    points = np.vstack((x, y, z)).T
                    cloud = pv.PolyData(points)
                    volume = cloud.delaunay_3d(alpha=tri_alpha)
                    shell = volume.extract_geometry()

                    x = shell.points[:, 0]
                    y = shell.points[:, 1]
                    z = shell.points[:, 2]
                    i = shell.faces.reshape((-1, 4))[:, 1]
                    j = shell.faces.reshape((-1, 4))[:, 2]
                    k = shell.faces.reshape((-1, 4))[:, 3]
                    indices = np.random.permutation(len(i))[:max_triangles]
                    i = i[indices]
                    j = j[indices]
                    k = k[indices]
                    triangles = np.vstack((i, j, k)).T

                    data.append(
                        go.Mesh3d(
                                x=x,
                                y=y,
                                z=z,
                                i=i,
                                j=j,
                                k=k,
                                opacity=surface_opacity,
                                color=color,
                                lighting=dict(ambient=0.9)
                        ))
                
            elif plot_type == 'points':
                
                pre_mask_xyz = mask_cxyz[mask_int]
                x = X[pre_mask_xyz].flatten()
                y = Y[pre_mask_xyz].flatten()
                z = Z[pre_mask_xyz].flatten()
                indices = np.random.permutation(len(x))[:surface_max_points]
                x = x[indices]
                y = y[indices]
                z = z[indices]
                
                data.append(
                    go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode='markers',
                            marker={
                                'size': point_cloud_size,
                                'color': color,
                                'opacity': point_cloud_opacity
                            }))
                

        fig = go.Figure(data);

    # setup the scene
    camera_dict = dict(
        up=dict(
            x=cam_props['cleft_normal'][0],
            y=cam_props['cleft_normal'][1],
            z=cam_props['cleft_normal'][2],
        ),
        eye=dict(
            x=zoom_out * (cam_props[f'cleft_inplane_{view_plane}'][0] - 0. * cam_props['cleft_normal'][0]),
            y=zoom_out * (cam_props[f'cleft_inplane_{view_plane}'][1] - 0. * cam_props['cleft_normal'][1]),
            z=zoom_out * (cam_props[f'cleft_inplane_{view_plane}'][2] - 0. * cam_props['cleft_normal'][2])
        )
    )

    scene_dict = dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(range=[np.min(X), np.max(X)], visible=False),
        yaxis=dict(range=[np.min(Y), np.max(Y)], visible=False),
        zaxis=dict(range=[np.min(Z), np.max(Z)], visible=False)
    )

    fig.update_layout(
        scene=scene_dict,
        scene_camera=camera_dict,
        autosize=False,
        width=fig_width,
        height=fig_height,
        showlegend=False,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4
        ));

    return fig


def interact_synapse_sections(
        ctx: SynapseVisContext,
        synapse_index: int,
        slider_axis=2,
        figsize=(4, 4)):
    
    intensity_bcxyz, mask_bcxyz = ctx.aug.augment_raw_data([ctx.synapse_dataset[synapse_index][1]])
    intensity_bcxyz = intensity_bcxyz.cpu().numpy()
    mask_bcxyz = mask_bcxyz.cpu().numpy()
    
    interactive_plot = interactive(
        lambda section_idx: plot_section_1_channel(
            intensity_bcxyz[0, 0, ...],
            mask_bcxyz[0, ...],
            slider_axis=slider_axis,
            section_idx=section_idx,
            figsize=figsize),
        section_idx=(0, intensity_bcxyz.shape[-1] - 1))
    
    return interactive_plot
    