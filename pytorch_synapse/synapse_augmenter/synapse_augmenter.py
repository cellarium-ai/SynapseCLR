import torch
import numpy as np

from synapse_augmenter import consts

from typing import Tuple, List, Union
from boltons.cacheutils import cachedmethod
from scipy.spatial.transform import Rotation
import cc3d

# for cachedmethod
_cache = dict()

# constants
GAUSSIAN_BLUR_KERNEL_SIZE = 3  # must be an odd number


@cachedmethod(_cache)
def get_volume_for_structure(radius: int):
    x = torch.arange(-radius, radius + 1)[:, None, None]
    y = torch.arange(-radius, radius + 1)[None, :, None]
    z = torch.arange(-radius, radius + 1)[None, None, :]
    return ((x.pow(2) + y.pow(2) + z.pow(2)) <= (radius ** 2)).sum().item()

@cachedmethod(_cache)
def get_hole_volume_l(radii: Tuple[int]) -> np.ndarray:
    return np.asarray([get_volume_for_structure(radius) for radius in radii])

@cachedmethod(_cache)
def get_corner_cutout_masks(size: int) -> np.ndarray:
    """Get corner cutout masks.
    
    .. note:: the first mask (index 0) is identity (i.e. no cutout).
    """
    assert size % 4 == 0
    ranges = [
        (0, size // 2),
        (size // 4, 3 * size // 2),
        (size // 2, size)]
    ignore_range_set = {(1, 1, 1)}
    n_ranges = len(ranges)
    n_masks = n_ranges ** 3 - len(ignore_range_set) + 1
    masks_jxyz = np.ones((n_masks, size, size, size), dtype=np.bool)
    i_mask = 1
    for i_x in range(n_ranges):
        for i_y in range(n_ranges):
            for i_z in range(n_ranges):
                if (i_x, i_y, i_z) in ignore_range_set:
                    continue
                masks_jxyz[
                    i_mask,
                    ranges[i_x][0]:ranges[i_x][1],
                    ranges[i_y][0]:ranges[i_y][1],
                    ranges[i_z][0]:ranges[i_z][1]] = 0
                i_mask += 1
    return masks_jxyz


@cachedmethod(_cache)
def get_gaussian_blur_kernels(sigmas: Tuple[float]) -> np.ndarray:
    kernel_half_size = (GAUSSIAN_BLUR_KERNEL_SIZE - 1) // 2
    x = torch.arange(-kernel_half_size, kernel_half_size + 1, dtype=torch.float)[:, None, None]
    y = torch.arange(-kernel_half_size, kernel_half_size + 1, dtype=torch.float)[None, :, None]
    z = torch.arange(-kernel_half_size, kernel_half_size + 1, dtype=torch.float)[None, None, :]
    r2_xyz = x.pow(2) + y.pow(2) + z.pow(2)
    
    kernel_11xyz_list = []
    for sigma in sigmas:
        kernel_xyz = torch.exp(- r2_xyz / (2. * sigma ** 2))
        kernel_xyz = kernel_xyz / kernel_xyz.sum()
        kernel_11xyz_list.append(kernel_xyz[None, None, ...])        
    kernel_r1xyz = torch.cat(kernel_11xyz_list, dim=0)
    return kernel_r1xyz.cpu().numpy()


class SynapseAugmenter:
    
    def __init__(self, **kwargs):
        
        self.norm_scale = kwargs['norm_scale']
        self.intensity_mean = kwargs['intensity_mean']
        self.intensity_std = kwargs['intensity_std']
        self.batch_mode = kwargs['batch_mode']
        self.debug_mode = kwargs['debug_mode']
        
        self.axial_to_sagittal_spacing = kwargs['axial_to_sagittal_spacing']
        
        self.enable_section_cutout = kwargs['enable_section_cutout']
        self.section_cutout_prob = kwargs['section_cutout_prob']
        self.cutout_poisson_rate = kwargs['cutout_poisson_rate']

        self.enable_sectional_intensity = kwargs['enable_sectional_intensity']
        self.sectional_intensity_intercept_min = kwargs['sectional_intensity_intercept_min']
        self.sectional_intensity_intercept_max = kwargs['sectional_intensity_intercept_max']
        self.sectional_intensity_slope_min = kwargs['sectional_intensity_slope_min']
        self.sectional_intensity_slope_max = kwargs['sectional_intensity_slope_max']
        self.sectional_intensity_gamma_min = kwargs['sectional_intensity_gamma_min']
        self.sectional_intensity_gamma_max = kwargs['sectional_intensity_gamma_max']
        
        self.enable_global_intensity = kwargs['enable_global_intensity']
        self.global_intensity_intercept_min = kwargs['global_intensity_intercept_min']
        self.global_intensity_intercept_max = kwargs['global_intensity_intercept_max']
        self.global_intensity_slope_min = kwargs['global_intensity_slope_min']
        self.global_intensity_slope_max = kwargs['global_intensity_slope_max']
        self.global_intensity_gamma_min = kwargs['global_intensity_gamma_min']
        self.global_intensity_gamma_max = kwargs['global_intensity_gamma_max']
        
        self.enable_pixel_noise = kwargs['enable_pixel_noise']
        self.global_additive_gaussian_noise_scale_min = kwargs['global_additive_gaussian_noise_scale_min']
        self.global_additive_gaussian_noise_scale_max = kwargs['global_additive_gaussian_noise_scale_max']
        self.global_multiplicative_lognormal_noise_scale_min = kwargs['global_multiplicative_lognormal_noise_scale_min']
        self.global_multiplicative_lognormal_noise_scale_max = kwargs['global_multiplicative_lognormal_noise_scale_max']
        
        self.enable_gaussian_blur = kwargs['enable_gaussian_blur']
        self.gaussian_blur_prob = kwargs['gaussian_blur_prob']
        self.gaussian_blur_sigmas = tuple(kwargs['gaussian_blur_sigmas'])
        
        self.enable_swiss_cheese = kwargs['enable_swiss_cheese']
        self.swiss_cheese_max_vol_fraction = kwargs['swiss_cheese_max_vol_fraction']
        self.swiss_cheese_radii = tuple(kwargs['swiss_cheese_radii'])
        
        self.enable_corner_cutout = kwargs['enable_corner_cutout']
        self.corner_cutout_prob = kwargs['corner_cutout_prob']
        self.n_corner_cutout_applications = kwargs['n_corner_cutout_applications']
        
        self.enable_random_cutout = kwargs['enable_random_cutout']
        self.random_cutout_prob = kwargs['random_cutout_prob']
        self.preserve_center_unmasked = kwargs['preserve_center_unmasked']
        self.cutout_size_min = kwargs['cutout_size_min']
        self.cutout_size_max = kwargs['cutout_size_max']
        self.n_random_cutout_applications = kwargs['n_random_cutout_applications']

        self.enable_random_periphery_cutout = kwargs['enable_random_periphery_cutout']
        self.periphery_cutout_center_max_displacement_fraction = kwargs['periphery_cutout_center_max_displacement_fraction']
        self.periphery_cutout_center_fraction_min = kwargs['periphery_cutout_center_fraction_min']
        self.periphery_cutout_center_fraction_max = kwargs['periphery_cutout_center_fraction_max']
        self.random_periphery_cutout_prob = kwargs['random_periphery_cutout_prob']            
            
        self.enable_displacement = kwargs['enable_displacement']
        self.enable_rotation = kwargs['enable_rotation']
        self.global_max_relative_displacement = kwargs['global_max_relative_displacement']
        
        self.enable_x_flip = kwargs['enable_x_flip']
        self.enable_y_flip = kwargs['enable_y_flip']
        self.enable_z_flip = kwargs['enable_z_flip']
        self.enable_xy_swap = kwargs['enable_xy_swap']
        
        self.enable_active_zone = kwargs['enable_active_zone']
        self.active_zone_masking_prob = kwargs['active_zone_masking_prob']
        self.connectivity = kwargs['connectivity']
        self.active_zone_inflation_radii = tuple(kwargs['active_zone_inflation_radii'])
        self.final_crop_size = kwargs['final_crop_size']
        self.final_img_size = kwargs['final_img_size']
        
        self.channel_organization = kwargs['channel_organization']
        
        if self.channel_organization == 'merged':
            self.n_intensity_output_channels = 1
        elif self.channel_organization == 'separate':
            self.n_intensity_output_channels = 2
        else:
            raise ValueError

        self.dtype = kwargs['dtype']
        self.device = kwargs['device']

    @staticmethod
    def apply_random_section_cutout(
            intensity_bcxyz: torch.Tensor,
            section_cutout_prob: float,
            cutout_poisson_rate: float):
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype
        batch_size = intensity_bcxyz.shape[0]
        n_total_sections = intensity_bcxyz.shape[-1]
        do_section_cutout_b = torch.distributions.Bernoulli(
            probs=torch.tensor(section_cutout_prob, device=device, dtype=dtype)).sample([batch_size]).type(torch.bool)
        cutout_bernoulli_prob_bz = torch.tensor(
            cutout_poisson_rate / n_total_sections,
            device=device, dtype=dtype).expand([batch_size, n_total_sections])
        cutout_bernoulli_prob_bz = cutout_bernoulli_prob_bz * do_section_cutout_b[:, None]
        cutout_sections_bz = torch.distributions.Bernoulli(cutout_bernoulli_prob_bz).sample().type(torch.bool)
        intensity_bcxyz.permute(0, 4, 1, 2, 3)[cutout_sections_bz] = 0.

    @staticmethod
    def apply_random_sectional_intensity_distortion(
            intensity_bcxyz: torch.Tensor,
            sectional_intensity_intercept_min: float,
            sectional_intensity_intercept_max: float,
            sectional_intensity_slope_min: float,
            sectional_intensity_slope_max: float,
            sectional_intensity_gamma_min: float,
            sectional_intensity_gamma_max: float):
        """Applies a sectional intensity distortion.

        .. note:: The following transformation is applied to every section:

                I_out = intercept + slope * I_in ^ gamma,

          where intercept, mean, and gamma are i.i.d random numbers for each section.

        .. note:: Masked values will remain masked.

        """

        batch_size = intensity_bcxyz.shape[0]
        n_sections = intensity_bcxyz.shape[-1]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        intercept_bz = (
            sectional_intensity_intercept_min +
            (sectional_intensity_intercept_max - sectional_intensity_intercept_min) *
                torch.rand((batch_size, n_sections), device=device, dtype=dtype))

        slope_bz = (
            sectional_intensity_slope_min +
            (sectional_intensity_slope_max - sectional_intensity_slope_min) *
                torch.rand((batch_size, n_sections), device=device, dtype=dtype))

        gamma_bz = (
            sectional_intensity_gamma_min +
            (sectional_intensity_gamma_max - sectional_intensity_gamma_min) *
                torch.rand((batch_size, n_sections), device=device, dtype=dtype))

        intensity_bcxyz \
            .pow_(gamma_bz[:, None, None, None, :]) \
            .mul_(slope_bz[:, None, None, None, :]) \
            .add_(intercept_bz[:, None, None, None, :]) \
            .clamp_(0., 1.)
    
    @staticmethod
    def apply_random_global_intensity_distortion(
            intensity_bcxyz: torch.Tensor,
            global_intensity_intercept_min: float,
            global_intensity_intercept_max: float,
            global_intensity_slope_min: float,
            global_intensity_slope_max: float,
            global_intensity_gamma_min: float,
            global_intensity_gamma_max: float):
        """Applies a global intensity distortion.

        .. note:: The following transformation is applied to all pixels:

                I_out = intercept + slope * I_in ^ gamma,

          where intercept, mean, and gamma are global random numbers for the
          entire volume.

        .. note:: Masked values will remain masked.

        """

        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        batch_size = intensity_bcxyz.shape[0]

        intercept_b = (
            global_intensity_intercept_min +
            (global_intensity_intercept_max - global_intensity_intercept_min) *
                torch.rand((batch_size,), device=device, dtype=dtype))

        slope_b = (
            global_intensity_slope_min +
            (global_intensity_slope_max - global_intensity_slope_min) *
                torch.rand((batch_size,), device=device, dtype=dtype))

        gamma_b = (
            global_intensity_gamma_min +
            (global_intensity_gamma_max - global_intensity_gamma_min) *
                torch.rand((batch_size,), device=device, dtype=dtype))

        intensity_bcxyz \
            .pow_(gamma_b[:, None, None, None, None]) \
            .mul_(slope_b[:, None, None, None, None]) \
            .add_(intercept_b[:, None, None, None, None]) \
            .clamp_(0., 1.)

    @staticmethod
    def apply_pixel_noise(
            intensity_bcxyz: torch.Tensor,
            global_additive_gaussian_noise_scale_min: float,
            global_additive_gaussian_noise_scale_max: float,
            global_multiplicative_lognormal_noise_scale_min: float,
            global_multiplicative_lognormal_noise_scale_max: float):

        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        batch_size = intensity_bcxyz.shape[0]

        additive_noise_scale_b = (
            global_additive_gaussian_noise_scale_min +
            (global_additive_gaussian_noise_scale_max -
             global_additive_gaussian_noise_scale_min) * torch.rand(
                (batch_size,), device=device, dtype=dtype))

        multiplicative_noise_scale_b = (
            global_multiplicative_lognormal_noise_scale_min +
            (global_multiplicative_lognormal_noise_scale_max -
             global_multiplicative_lognormal_noise_scale_min) * torch.rand(
                (batch_size,), device=device, dtype=dtype))

        additive_noise_bcxyz = additive_noise_scale_b[:, None, None, None, None] * torch.randn_like(intensity_bcxyz)
        multiplicative_noise_bcxyz = torch.exp(
            multiplicative_noise_scale_b[:, None, None, None, None] * torch.randn_like(intensity_bcxyz))
        multiplicative_noise_bcxyz = (
            multiplicative_noise_bcxyz /
            multiplicative_noise_bcxyz.mean(dim=(-1, -2, -3, -4))[:, None, None, None, None])

        intensity_bcxyz \
            .mul_(multiplicative_noise_bcxyz) \
            .add_(additive_noise_bcxyz) \
            .clamp_(0., 1.)

    @staticmethod
    def get_random_3d_affine_matrix_batch(
            batch_size: int,
            batch_mode: str,
            enable_displacement: bool,
            enable_rotation: bool,
            global_max_relative_displacement: float,
            scale_factors_i: np.ndarray,
            device: torch.device,
            dtype: torch.dtype) -> torch.Tensor:
        """
        .. note:: only 3D rotations and translation; no scaling
        """
        
        if batch_mode == 'coupled':
            _batch_size = batch_size
            batch_size = 1
        
        affine_bik = torch.zeros((batch_size, 3, 4), device=device, dtype=dtype)

        if enable_rotation:
            # random 3D rotation
            rot_matrices_bij = torch.tensor(
                Rotation.random(batch_size).as_matrix(), device=device, dtype=dtype)
        else:
            rot_matrices_bij = torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3).contiguous()

        # scale factors along the 3 axes
        for i in range(3):
            rot_matrices_bij[:, i, :] = rot_matrices_bij[:, i, :] / scale_factors_i[i]

        # random displacement of the origin
        if enable_displacement:
            r_bi = 2. * torch.rand((batch_size, 3), device=device, dtype=dtype) - 1.
            r_bi = global_max_relative_displacement * r_bi
        else:
            r_bi = torch.zeros((batch_size, 3), device=device, dtype=dtype)

        affine_bik[:, :, :3] = rot_matrices_bij
        affine_bik[:, :, 3] = r_bi

        if batch_mode == 'coupled':
            affine_bik = affine_bik.expand((_batch_size, 3, 4))
            
        return affine_bik

    @staticmethod
    def apply_random_3d_affine_transformation_to_volume(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            enable_displacement: bool,
            enable_rotation: bool,
            final_crop_size: int,
            final_img_size: int,
            axial_to_sagittal_spacing: int,
            global_max_relative_displacement: float,
            batch_mode: str,
            align_corners: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """TBW."""

        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        batch_size, n_channels, orig_x_size, orig_y_size, orig_z_size = intensity_bcxyz.shape
        assert orig_x_size == orig_y_size

        # global scale factor is for cropping
        global_scale_factor = orig_x_size / final_crop_size
        
        # per_axis_scale_factor_i is for correcting for too many/few sections
        z_scale_factor = (axial_to_sagittal_spacing * orig_z_size) / orig_x_size
        per_axis_scale_factor_i = np.asarray([1., 1., z_scale_factor])

        scale_factors_i = global_scale_factor * per_axis_scale_factor_i
        final_shape = (
            batch_size, n_channels,
            final_img_size, final_img_size, final_img_size)

        # generate random affine theta
        theta_bik = SynapseAugmenter.get_random_3d_affine_matrix_batch(
            batch_size, batch_mode, enable_displacement, enable_rotation,
            global_max_relative_displacement, scale_factors_i, device, dtype)

        # make the affine grid
        grid = torch.nn.functional.affine_grid(
            theta_bik, final_shape, align_corners=align_corners)

        # transform intensity
        affine_intensity_bcxyz = torch.nn.functional.grid_sample(
            intensity_bcxyz, grid, align_corners=align_corners)
        affine_intensity_bcxyz.clamp_(0., 1.)

        # transform mask
        affine_mask_bcxyz = torch.nn.functional.grid_sample(
            mask_bcxyz.to(dtype), grid, align_corners=align_corners) > 0.5

        return affine_intensity_bcxyz, affine_mask_bcxyz

    @staticmethod
    def inflate_binary_mask(mask_bxyz: torch.Tensor, radius: int) -> torch.Tensor:
        assert radius >= 0
        
        if radius == 0:
            return mask_bxyz
        
        device = mask_bxyz.device
        
        x = torch.arange(-radius, radius + 1, dtype=torch.float, device=device)[:, None, None]
        y = torch.arange(-radius, radius + 1, dtype=torch.float, device=device)[None, :, None]
        z = torch.arange(-radius, radius + 1, dtype=torch.float, device=device)[None, None, :]
        
        struct = ((x.pow(2) + y.pow(2) + z.pow(2)) <= (radius ** 2)).float()
        kern = struct[None, None, ...]
  
        return (torch.nn.functional.conv3d(
            mask_bxyz[:, None, :, :, :].float(), kern, padding=radius) > 0)[:, 0, :, :, :].type(torch.bool)

    @staticmethod
    def get_active_zone_binary_mask(
            mask_bcxyz: torch.Tensor,
            connectivity: int,
            active_zone_inflation_radii: Tuple[int],
            batch_mode: str) -> Tuple[torch.Tensor, int]:
        """
        
        .. note:: this method always has a "coupled" behavior
        
        """

        assert mask_bcxyz.ndim == 5
        batch_size = mask_bcxyz.shape[0]
        n_radii = len(active_zone_inflation_radii)        
        device = mask_bcxyz.device
        
        # bitwise_or of all non-background pixels
        mask_bcxyz_np = mask_bcxyz.cpu().numpy()
        pre_and_cleft_binary_mask_bxyz_np = (
            mask_bcxyz_np[:, consts.MASK_PRE_SYNAPTIC_NEURON - 1] |
            mask_bcxyz_np[:, consts.MASK_SYNAPTIC_CLEFT - 1])
        post_and_cleft_binary_mask_bxyz_np = (
            mask_bcxyz_np[:, consts.MASK_POST_SYNAPTIC_NEURON - 1] |
            mask_bcxyz_np[:, consts.MASK_SYNAPTIC_CLEFT - 1])
        cleft_binary_mask_bxyz_np = mask_bcxyz_np[:, consts.MASK_SYNAPTIC_CLEFT - 1]
                
        active_zone_mask_1xyz_np_list = []
        for i_batch in range(batch_size):

            if connectivity > 0:
                
                pre_and_cleft_binary_mask_xyz_np = pre_and_cleft_binary_mask_bxyz_np[i_batch]
                post_and_cleft_binary_mask_xyz_np = post_and_cleft_binary_mask_bxyz_np[i_batch]
                cleft_mask_xyz_np = cleft_binary_mask_bxyz_np[i_batch]

                pre_and_cleft_labels_xyz_np = cc3d.connected_components(
                    pre_and_cleft_binary_mask_xyz_np,
                    connectivity=connectivity)
                post_and_cleft_labels_xyz_np = cc3d.connected_components(
                    post_and_cleft_binary_mask_xyz_np,
                    connectivity=connectivity)

                # identify all labels that overlap with the cleft
                pre_cleft_overlapping_labels = np.unique(
                    (pre_and_cleft_labels_xyz_np * cleft_mask_xyz_np)[cleft_mask_xyz_np > 0])
                post_cleft_overlapping_labels = np.unique(
                    (post_and_cleft_labels_xyz_np * cleft_mask_xyz_np)[cleft_mask_xyz_np > 0])

                active_zone_mask_xyz_np = np.zeros_like(cleft_mask_xyz_np)
                for component_label in pre_cleft_overlapping_labels:
                    active_zone_mask_xyz_np = (
                        active_zone_mask_xyz_np |
                        (pre_and_cleft_labels_xyz_np == component_label))
                for component_label in post_cleft_overlapping_labels:
                    active_zone_mask_xyz_np = (
                        active_zone_mask_xyz_np |
                        (post_and_cleft_labels_xyz_np == component_label))
                    
            else:  # do not do connected components
                active_zone_mask_xyz_np = np.sum(mask_bcxyz_np[i_batch], 0) > 0
                

            # add to list
            active_zone_mask_1xyz_np_list.append(active_zone_mask_xyz_np[None, ...])

        # inflate
        active_zone_inflation_radius_idx = torch.distributions.Categorical(
            torch.ones(n_radii, device=device, dtype=torch.float32)).sample([1]).item()
        active_zone_inflation_radius = active_zone_inflation_radii[active_zone_inflation_radius_idx]

        active_zone_mask_bxyz = SynapseAugmenter.inflate_binary_mask(
            torch.tensor(
                np.concatenate(active_zone_mask_1xyz_np_list, axis=0),
                device=mask_bcxyz.device, dtype=torch.bool), 
            radius=active_zone_inflation_radius)

        return active_zone_mask_bxyz, active_zone_inflation_radius
    
    @staticmethod
    def apply_swiss_cheese_hollow_out(
            intensity_bcxyz: torch.Tensor,
            swiss_cheese_max_vol_fraction: float,
            swiss_cheese_radii: Tuple[int],
            batch_mode: str):

        assert all(radius > 0 for radius in swiss_cheese_radii)
        assert swiss_cheese_max_vol_fraction >= 0.

        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype
        batch_size = intensity_bcxyz.shape[0]

        if batch_mode == 'coupled':
            _batch_size = batch_size
            batch_size = 1
            
        # sample to-be-hollowed-out volume fraction for the batch
        vol_fraction_b = swiss_cheese_max_vol_fraction * torch.rand(
            batch_size, dtype=dtype, device=device)

        # assuming uniform distribution of holes of various sizes, sample
        # a Bernoulli field for the position of the holes for each element
        # in the batch
        total_volume = intensity_bcxyz.shape[-1] * intensity_bcxyz.shape[-2] * intensity_bcxyz.shape[-3]
        hole_volume_l = torch.tensor(
            get_hole_volume_l(swiss_cheese_radii),
            device=device, 
            dtype=dtype)
        
        hole_count_bl = vol_fraction_b[:, None] * total_volume / (hole_volume_l * hole_volume_l.shape[0])
        bernoulli_prob_bl = hole_count_bl / total_volume
        hole_indicator_blxyz = torch.distributions.Bernoulli(
            probs=bernoulli_prob_bl[:, :, None, None, None].expand(
                hole_count_bl.shape + intensity_bcxyz.shape[-3:])).sample([])

        # inflate the hole indicators by their respective radii and reduce to a single mask
        swiss_cheese_bxyz = torch.sum(
            torch.cat([
                SynapseAugmenter.inflate_binary_mask(hole_indicator_blxyz[:, l, ...], radius)[:, None, ...]
                for l, radius in enumerate(swiss_cheese_radii)],
                dim=1),
            dim=1) > 0

        if batch_mode == 'coupled':
            swiss_cheese_bxyz = swiss_cheese_bxyz.expand((_batch_size,) + intensity_bcxyz.shape[-3:])

        # apply swiss cheese in-place
        intensity_bcxyz.mul_(~swiss_cheese_bxyz[:, None, ...])

    @staticmethod
    def apply_corner_cutout(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            corner_cutout_prob: float,
            preserve_center_unmasked: bool,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        size = intensity_bcxyz.shape[-1]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype
        
        if batch_mode == 'coupled':
            _batch_size = batch_size
            batch_size = 1
            
        masks_jxyz = torch.tensor(get_corner_cutout_masks(size), device=device, dtype=torch.bool)
        n_masks = masks_jxyz.shape[0]
        do_corner_cutout_b = torch.distributions.Bernoulli(
            probs=torch.tensor(corner_cutout_prob, device=device, dtype=dtype)).sample([batch_size]).type(torch.bool)
        corner_index_b = torch.distributions.Categorical(
            probs=torch.ones(n_masks - 1, device=device, dtype=dtype)).sample([batch_size]) + 1
        corner_index_b[~do_corner_cutout_b] = 0
        masks_bxyz = masks_jxyz[corner_index_b, :, :, :]
        if preserve_center_unmasked:
            center_lo = size // 4
            center_hi = 3 * size // 4
            masks_bxyz[:, center_lo:center_hi, center_lo:center_hi, center_lo:center_hi] = 1
        
        if batch_mode == 'coupled':
            masks_bxyz = masks_bxyz.expand((_batch_size,) + intensity_bcxyz.shape[-3:])

        intensity_bcxyz.mul_(masks_bxyz[:, None, :, :, :])
        mask_bcxyz.mul_(masks_bxyz[:, None, :, :, :])
    
    @staticmethod
    def apply_random_periphery_cutout(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            center_max_displacement_fraction: float,
            center_fraction_min: float,
            center_fraction_max: float,
            random_periphery_cutout_prob: bool,
            batch_mode: str):
        
        assert 0. < center_max_displacement_fraction < 1.
        assert 0. < center_fraction_min < 1.
        assert 0. < center_fraction_max < 1.
        assert 0. <= random_periphery_cutout_prob <= 1.
        
        batch_size = intensity_bcxyz.shape[0]
        size = intensity_bcxyz.shape[-1]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype
        
        if batch_mode == 'coupled':
            _batch_size = batch_size
            batch_size = 1
        
        center_max_displacement_size = 0.5 * center_max_displacement_fraction * size
        
        c_x_b = torch.randint(
            max(int(0.5 * size - center_max_displacement_size), 0),
            min(int(0.5 * size + center_max_displacement_size), size),
            [batch_size], device=device)
        c_y_b = torch.randint(
            max(int(0.5 * size - center_max_displacement_size), 0),
            min(int(0.5 * size + center_max_displacement_size), size),
            [batch_size], device=device)
        c_z_b = torch.randint(
            max(int(0.5 * size - center_max_displacement_size), 0),
            min(int(0.5 * size + center_max_displacement_size), size),
            [batch_size], device=device)
        half_center_size_b = torch.randint(
            int(0.5 * size * center_fraction_min),
            int(0.5 * size * center_fraction_max),
            [batch_size], device=device)
        i_x_b = torch.maximum(c_x_b - half_center_size_b, torch.tensor(0, device=device))
        i_y_b = torch.maximum(c_y_b - half_center_size_b, torch.tensor(0, device=device))
        i_z_b = torch.maximum(c_z_b - half_center_size_b, torch.tensor(0, device=device))
        j_x_b = torch.minimum(c_x_b + half_center_size_b, torch.tensor(size, device=device))
        j_y_b = torch.minimum(c_y_b + half_center_size_b, torch.tensor(size, device=device))
        j_z_b = torch.minimum(c_z_b + half_center_size_b, torch.tensor(size, device=device))
        
        do_random_cutout_b = torch.distributions.Bernoulli(
            probs=torch.tensor(random_periphery_cutout_prob, device=device, dtype=dtype)).sample([batch_size]).type(torch.bool)
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_random_cutout_b]
        mask_bxyz = torch.ones((batch_size, size, size, size), device=device, dtype=torch.bool)

        for i_batch in do_batch_indices:
            mask_bxyz[i_batch, ...] = 0
            mask_bxyz[
                i_batch,
                i_x_b[i_batch]:j_x_b[i_batch],
                i_y_b[i_batch]:j_y_b[i_batch],
                i_z_b[i_batch]:j_z_b[i_batch]] = 1
                        
        if batch_mode == 'coupled':
            mask_bxyz = mask_bxyz.expand((_batch_size,) + intensity_bcxyz.shape[-3:])

        intensity_bcxyz.mul_(mask_bxyz[:, None, :, :, :])
        mask_bcxyz.mul_(mask_bxyz[:, None, :, :, :])

    @staticmethod
    def apply_random_cutout(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            preserve_center_unmasked: bool,
            cutout_size_min: int,
            cutout_size_max: int,
            random_cutout_prob: bool,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        size = intensity_bcxyz.shape[-1]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype
        
        if batch_mode == 'coupled':
            _batch_size = batch_size
            batch_size = 1
            
        assert cutout_size_min > 0
        assert cutout_size_max > 0
        assert cutout_size_max >= cutout_size_min
        assert cutout_size_max < size
        
        i_x_b = torch.randint(0, size - cutout_size_min, [batch_size], device=device)
        i_y_b = torch.randint(0, size - cutout_size_min, [batch_size], device=device)
        i_z_b = torch.randint(0, size - cutout_size_min, [batch_size], device=device)
        j_x_b = torch.minimum(
            i_x_b + torch.randint(cutout_size_min, cutout_size_max + 1, [batch_size], device=device),
            torch.tensor(size, device=device))
        j_y_b = torch.minimum(
            i_y_b + torch.randint(cutout_size_min, cutout_size_max + 1, [batch_size], device=device),
            torch.tensor(size, device=device))
        j_z_b = torch.minimum(
            i_z_b + torch.randint(cutout_size_min, cutout_size_max + 1, [batch_size], device=device),
            torch.tensor(size, device=device))
        do_random_cutout_b = torch.distributions.Bernoulli(
            probs=torch.tensor(random_cutout_prob, device=device, dtype=dtype)).sample([batch_size]).type(torch.bool)
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_random_cutout_b]
        mask_bxyz = torch.ones((batch_size, size, size, size), device=device, dtype=torch.bool)
        for i_batch in do_batch_indices:
            mask_bxyz[
                i_batch,
                i_x_b[i_batch]:j_x_b[i_batch],
                i_y_b[i_batch]:j_y_b[i_batch],
                i_z_b[i_batch]:j_z_b[i_batch]] = 0
        if preserve_center_unmasked:
            center_lo = size // 4
            center_hi = 3 * size // 4
            mask_bxyz[:, center_lo:center_hi, center_lo:center_hi, center_lo:center_hi] = 1
            
        if batch_mode == 'coupled':
            mask_bxyz = mask_bxyz.expand((_batch_size,) + intensity_bcxyz.shape[-3:])

        intensity_bcxyz.mul_(mask_bxyz[:, None, :, :, :])
        mask_bcxyz.mul_(mask_bxyz[:, None, :, :, :])

    @staticmethod
    def apply_x_flip(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        x_size, y_size, z_size = intensity_bcxyz.shape[2:]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        x_reverse_range = torch.arange(x_size - 1, -1, -1, device=device, dtype=torch.long)

        if batch_mode == 'decoupled':
            do_flip_b = torch.rand(batch_size, device=device) > 0.5
        else:
            do_flip_b = (torch.rand(1, device=device) > 0.5).expand((batch_size,))
            
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_flip_b]
        
        for i_batch in do_batch_indices:
            intensity_bcxyz[i_batch, ...] = intensity_bcxyz[i_batch, :, x_reverse_range, :, :]
            mask_bcxyz[i_batch, ...] = mask_bcxyz[i_batch, :, x_reverse_range, :, :]

    @staticmethod
    def apply_y_flip(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        x_size, y_size, z_size = intensity_bcxyz.shape[2:]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        y_reverse_range = torch.arange(y_size - 1, -1, -1, device=device, dtype=torch.long)

        if batch_mode == 'decoupled':
            do_flip_b = torch.rand(batch_size, device=device) > 0.5
        else:
            do_flip_b = (torch.rand(1, device=device) > 0.5).expand((batch_size,))
        
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_flip_b]
        
        for i_batch in do_batch_indices:
            intensity_bcxyz[i_batch, ...] = intensity_bcxyz[i_batch, :, :, y_reverse_range, :]
            mask_bcxyz[i_batch, ...] = mask_bcxyz[i_batch, :, :, y_reverse_range, :]

    @staticmethod
    def apply_z_flip(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        x_size, y_size, z_size = intensity_bcxyz.shape[2:]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        z_reverse_range = torch.arange(z_size - 1, -1, -1, device=device, dtype=torch.long)

        if batch_mode == 'decoupled':
            do_flip_b = torch.rand(batch_size, device=device) > 0.5
        else:
            do_flip_b = (torch.rand(1, device=device) > 0.5).expand((batch_size,))
        
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_flip_b]
        
        for i_batch in do_batch_indices:
            intensity_bcxyz[i_batch, ...] = intensity_bcxyz[i_batch, :, :, :, z_reverse_range]
            mask_bcxyz[i_batch, ...] = mask_bcxyz[i_batch, :, :, :, z_reverse_range]

    @staticmethod
    def apply_xy_swap(
            intensity_bcxyz: torch.Tensor,
            mask_bcxyz: torch.Tensor,
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        if batch_mode == 'decoupled':
            do_swap_b = torch.rand(batch_size, device=device) > 0.5
        else:
            do_swap_b = (torch.rand(1, device=device) > 0.5).expand((batch_size,))

        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_swap_b]
        
        intensity_bcxyz[do_batch_indices, ...] = intensity_bcxyz[do_batch_indices, ...].permute(0, 1, 3, 2, 4).clone()
        mask_bcxyz[do_batch_indices, ...] = mask_bcxyz[do_batch_indices, ...].permute(0, 1, 3, 2, 4).clone()
            
    @staticmethod
    def apply_gaussian_blur(
            intensity_bcxyz: torch.Tensor,
            gaussian_blur_prob: float,
            gaussian_blur_sigmas: Tuple[float],
            batch_mode: str):
        
        batch_size = intensity_bcxyz.shape[0]
        n_input_channels = intensity_bcxyz.shape[1]
        device = intensity_bcxyz.device
        dtype = intensity_bcxyz.dtype

        # select samples to actually apply blur on
        if batch_mode == 'decoupled':
            do_blur_b = torch.rand(batch_size, device=device) < gaussian_blur_prob
        else:
            do_blur_b = (torch.rand(1, device=device) < gaussian_blur_prob).expand((batch_size,))
        
        do_batch_indices = torch.arange(0, batch_size, device=device, dtype=torch.long)[do_blur_b]
        n_blur_apply = do_batch_indices.shape[0]
        
        if n_blur_apply == 0:
            return
        
        kern_r1xyz = torch.tensor(get_gaussian_blur_kernels(gaussian_blur_sigmas), device=device, dtype=dtype)
        n_sigmas = kern_r1xyz.shape[0]
        kernel_size = kern_r1xyz.shape[-1]
        
        # select blur sigmas for each sample
        if batch_mode == 'decoupled':
            blur_radius_idx_b = torch.distributions.Categorical(
                torch.ones(n_sigmas, device=device, dtype=dtype)).sample([batch_size])
        else:
            blur_radius_idx_b = torch.distributions.Categorical(
                torch.ones(n_sigmas, device=device, dtype=dtype)).sample([1]).expand([batch_size])
        
        # we blur only the selected batch indices
        for i_input_channel in range(n_input_channels):
            blurred_intensity_drxyz = torch.nn.functional.conv3d(
                intensity_bcxyz[do_batch_indices, i_input_channel, ...][:, None, ...],
                kern_r1xyz,
                padding=(kernel_size - 1) // 2)
            intensity_bcxyz[do_batch_indices, i_input_channel, ...] = blurred_intensity_drxyz[
                torch.arange(0, n_blur_apply), blur_radius_idx_b[do_batch_indices], ...]        
        
    def __call__(
            self,
            intensity_mask_bcxyz_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # generate intensity tensor
        intensity_bcxyz = torch.clamp(
            torch.tensor(
                intensity_mask_bcxyz_np[:, consts.INPUT_DATA_INTENSITY_CHANNEL, ...],
                device=self.device, dtype=self.dtype) / self.norm_scale,
            min=0., max=1.)[:, None, ...]
        
        # generate binary mask tensor
        integer_mask_bxyz = torch.tensor(
            intensity_mask_bcxyz_np[:, consts.INPUT_DATA_MASK_CHANNEL, ...],
            device=self.device)
        mask_bcxyz = torch.cat(
            [(integer_mask_bxyz == mask_int_value)[:, None, ...]
             for mask_int_value in consts.MASK_INTEGER_VALUES], dim=-4)
        
        batch_size = intensity_bcxyz.shape[0]
        
        if self.enable_section_cutout:
            self.apply_random_section_cutout(
                intensity_bcxyz,
                section_cutout_prob=self.section_cutout_prob,
                cutout_poisson_rate=self.cutout_poisson_rate)

        if self.enable_sectional_intensity:
            self.apply_random_sectional_intensity_distortion(
                intensity_bcxyz,
                sectional_intensity_intercept_min=self.sectional_intensity_intercept_min,
                sectional_intensity_intercept_max=self.sectional_intensity_intercept_max,
                sectional_intensity_slope_min=self.sectional_intensity_slope_min,
                sectional_intensity_slope_max=self.sectional_intensity_slope_max,
                sectional_intensity_gamma_min=self.sectional_intensity_gamma_min,
                sectional_intensity_gamma_max=self.sectional_intensity_gamma_max)

        if self.enable_global_intensity:
            self.apply_random_global_intensity_distortion(
                intensity_bcxyz,
                global_intensity_intercept_min=self.global_intensity_intercept_min,
                global_intensity_intercept_max=self.global_intensity_intercept_max,
                global_intensity_slope_min=self.global_intensity_slope_min,
                global_intensity_slope_max=self.global_intensity_slope_max,
                global_intensity_gamma_min=self.global_intensity_gamma_min,
                global_intensity_gamma_max=self.global_intensity_gamma_max)

        if self.enable_pixel_noise:
            self.apply_pixel_noise(
                intensity_bcxyz,
                global_additive_gaussian_noise_scale_min=self.global_additive_gaussian_noise_scale_min,
                global_additive_gaussian_noise_scale_max=self.global_additive_gaussian_noise_scale_max,
                global_multiplicative_lognormal_noise_scale_min=self.global_multiplicative_lognormal_noise_scale_min,
                global_multiplicative_lognormal_noise_scale_max=self.global_multiplicative_lognormal_noise_scale_max)
        
        intensity_bcxyz, mask_bcxyz = self.apply_random_3d_affine_transformation_to_volume(
            intensity_bcxyz,
            mask_bcxyz,
            enable_displacement=self.enable_displacement,
            enable_rotation=self.enable_rotation,
            final_crop_size=self.final_crop_size,
            final_img_size=self.final_img_size,
            axial_to_sagittal_spacing=self.axial_to_sagittal_spacing,
            global_max_relative_displacement=self.global_max_relative_displacement,
            batch_mode=self.batch_mode)
        
        if self.enable_gaussian_blur:
            self.apply_gaussian_blur(
                intensity_bcxyz,
                gaussian_blur_prob=self.gaussian_blur_prob,
                gaussian_blur_sigmas=self.gaussian_blur_sigmas,
                batch_mode=self.batch_mode)      
        
        if self.enable_swiss_cheese:
            self.apply_swiss_cheese_hollow_out(
                intensity_bcxyz,
                swiss_cheese_max_vol_fraction=self.swiss_cheese_max_vol_fraction,
                swiss_cheese_radii=self.swiss_cheese_radii,
                batch_mode=self.batch_mode)
            
        if self.enable_corner_cutout:
            for _ in range(self.n_corner_cutout_applications):
                self.apply_corner_cutout(
                    intensity_bcxyz,
                    mask_bcxyz,
                    corner_cutout_prob=self.corner_cutout_prob,
                    preserve_center_unmasked=self.preserve_center_unmasked,
                    batch_mode=self.batch_mode)

        if self.enable_random_cutout:
            for _ in range(self.n_random_cutout_applications):
                self.apply_random_cutout(
                    intensity_bcxyz,
                    mask_bcxyz,
                    preserve_center_unmasked=self.preserve_center_unmasked,
                    cutout_size_min=self.cutout_size_min,
                    cutout_size_max=self.cutout_size_max,
                    random_cutout_prob=self.random_cutout_prob,
                    batch_mode=self.batch_mode)
                 
        if self.enable_random_periphery_cutout:
            self.apply_random_periphery_cutout(
                intensity_bcxyz=intensity_bcxyz,
                mask_bcxyz=mask_bcxyz,
                center_max_displacement_fraction=self.periphery_cutout_center_max_displacement_fraction,
                center_fraction_min=self.periphery_cutout_center_fraction_min,
                center_fraction_max=self.periphery_cutout_center_fraction_max,
                random_periphery_cutout_prob=self.random_periphery_cutout_prob,
                batch_mode=self.batch_mode)

        if self.enable_x_flip:
            self.apply_x_flip(intensity_bcxyz, mask_bcxyz, batch_mode=self.batch_mode)
            
        if self.enable_y_flip:
            self.apply_y_flip(intensity_bcxyz, mask_bcxyz, batch_mode=self.batch_mode)

        if self.enable_z_flip:
            self.apply_z_flip(intensity_bcxyz, mask_bcxyz, batch_mode=self.batch_mode)
            
        if self.enable_xy_swap:
            self.apply_xy_swap(intensity_bcxyz, mask_bcxyz, batch_mode=self.batch_mode)

        if self.enable_active_zone:

            # get the active zone
            active_zone_mask_bxyz, chosen_inflation_radius = self.get_active_zone_binary_mask(
                mask_bcxyz,
                connectivity=self.connectivity,
                active_zone_inflation_radii=self.active_zone_inflation_radii,
                batch_mode=self.batch_mode)
            
        else:
            
            active_zone_mask_bxyz = torch.ones(
                (batch_size,) + intensity_bcxyz.shape[2:], device=intensity_bcxyz.device, dtype=torch.bool)
            
            chosen_inflation_radius = 0

        if self.channel_organization == 'merged':
            
            # cut both the mask and the intensity tensor with the active zone mask
            mask_bcxyz.mul_(active_zone_mask_bxyz[:, None, ...])
            intensity_bcxyz.mul_(active_zone_mask_bxyz[:, None, ...])
            
        elif self.channel_organization == 'separate':
            
            pre_and_cleft_binary_mask_bxyz = SynapseAugmenter.inflate_binary_mask(
                mask_bcxyz[:, consts.MASK_PRE_SYNAPTIC_NEURON - 1] | mask_bcxyz[:, consts.MASK_SYNAPTIC_CLEFT - 1],
                radius=chosen_inflation_radius) & active_zone_mask_bxyz
            
            post_and_cleft_binary_mask_bxyz = SynapseAugmenter.inflate_binary_mask(
                mask_bcxyz[:, consts.MASK_POST_SYNAPTIC_NEURON - 1] | mask_bcxyz[:, consts.MASK_SYNAPTIC_CLEFT - 1],
                radius=chosen_inflation_radius) & active_zone_mask_bxyz

            intensity_bcxyz = torch.cat([
                (intensity_bcxyz[:, 0, ...] * pre_and_cleft_binary_mask_bxyz)[:, None, ...],
                (intensity_bcxyz[:, 0, ...] * post_and_cleft_binary_mask_bxyz)[:, None, ...]], dim=1)
        
        else:
            
            raise ValueError
            
        # z-score intensities
        intensity_bcxyz = (intensity_bcxyz - self.intensity_mean) / self.intensity_std

        if self.debug_mode:
            # print the random state
            print(f'pytorch random: {torch.rand(1).item():.3f}, numpy random: {np.random.rand():.3f}')
        
        return intensity_bcxyz, mask_bcxyz
    
    def augment_raw_data(
            self,
            loaded_npy_data: Union[List[np.ndarray], np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # basic check of input data
        if isinstance(loaded_npy_data, np.ndarray):
            loaded_npy_data = [loaded_npy_data]
        elif isinstance(loaded_npy_data, list):
            assert all(isinstance(a, np.ndarray) for a in loaded_npy_data)
        else:
            raise ValueError
        assert all(a.ndim == 4 for a in loaded_npy_data)
        assert all(a.shape[0] == 2 for a in loaded_npy_data)
        
        intensity_mask_bcxyz = np.concatenate(
            [intensity_mask_cxyz[None, ...] for intensity_mask_cxyz in loaded_npy_data],
            axis=0)
        
        return self(intensity_mask_bcxyz)
