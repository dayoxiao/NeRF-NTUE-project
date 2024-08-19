import torch
from torch import nn

import numpy as np

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def sample_stratified(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  n_samples: int,
  perturb: bool = True,
  inverse_depth: bool = False
):
  """
  Sample points along rays, stage 1
  Return: Tuple[torch.Tensor, torch.Tensor]
  """

  # Grab n samples from 0 to 1, for space integration along ray
  t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)

  if not inverse_depth:
    # Sample linearly between near and far
    z_vals = near * (1.-t_vals) + far * (t_vals)
  else:
    # Use inverse depth
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

  # Draw samples along ray
  if perturb:
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[...,-1:]], dim=-1)
    lower = torch.cat([z_vals[...,:1], mids], dim=-1)
    t_rand = torch.rand([n_samples], device=z_vals.device)
    z_vals = lower + (upper - lower) * t_rand
  z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

  # Apply scale from rays_d and offset from rays_o to samples
  # [10000, 1, 3] + [10000, 1, 3] * [10000, 8, 1] = [10000, 8, 3]
  pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
  return pts, z_vals
  
def raw2outputs(
  raw: torch.Tensor,
  z_vals: torch.Tensor,
  rays_d: torch.Tensor,
  raw_noise_std: float = 0.,
  white_bkgd: bool = False
):
  """
  Volume rendering, convert raw NeRF output into RGB and other maps.
  Return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
  """

  # Distance between each point
  deltas = z_vals[..., 1:] - z_vals[..., :-1]
  delta_last = 1e10 * torch.ones_like(deltas[..., :1])
  deltas = torch.cat([deltas, delta_last], dim=-1)

  # Multiply each distance by the norm to convert to real world distance
  deltas = deltas * torch.norm(rays_d[..., None, :], dim=-1)

  # Add noise
  noise = 0.
  if raw_noise_std > 0.:
    noise = torch.randn(raw[..., 3].shape) * raw_noise_std

  # Predict density of each sample along each ray. [n_rays, n_samples]
  alphas = 1.0 - torch.exp(-deltas * nn.functional.relu(raw[..., 3] + noise))
  alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1-alphas+1e-10], dim=-1)

  # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
  weights = alphas * torch.cumprod(alphas_shifted, -1)[..., :-1]

  # Compute weighted RGB map.
  rgbs = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
  rgb_map = torch.sum(weights[..., None] * rgbs, dim=-2)  # [n_rays, 3]

  # Depth map (depth)
  depth_map = torch.sum(weights * z_vals, dim=-1)

  # Disparity map (inverse depth)
  disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))

  # Sum of weights along each ray. [0, 1] up to numerical error.
  weights_sum = torch.sum(weights, dim=-1)

  # If white background, use the accumulated alpha map.
  if white_bkgd:
    rgb_map = rgb_map + (1. - weights_sum[..., None])

  return rgb_map, disp_map, weights_sum, weights, depth_map

def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_importance: int,
  perturb: bool = False
):
  """
  Index for Hierarchical Volume Sampling (stage 2)
  Return: torch.Tensor
  """

  N_rays, N_samples = weights.shape
  weights = weights + 1e-5    # prevent nans

  # Get PDF.
  pdf = weights / torch.sum(weights, -1, keepdims=True) # [n_rays, N_samples]

  # Convert PDF to CDF.
  cdf = torch.cumsum(pdf, dim=-1) # [n_rays, N_samples]
  cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, N_samples+1]

  # Take sample positions to grab from CDF. Linear or perturbed
  if not perturb:
    u = torch.linspace(0., 1., n_importance, device=cdf.device)
    u = u.expand([N_rays] + [n_importance]) # [n_rays, n_importance]
  else:
    u = torch.rand([N_rays] + [n_importance], device=cdf.device) # [n_rays, n_importance]

  # Find indexes along CDF where values in u would be placed.
  u = u.contiguous() # Returns contiguous tensor with same values.
  inds = torch.searchsorted(cdf, u, side="right") # [n_rays, n_importance]

  # Clamp indexes out of bounds.
  below = torch.clamp_min(inds-1, min=0)
  above = torch.clamp_max(inds, max=N_samples)
  inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_importance, 2]
  inds_sampled = inds_g.view(N_rays, 2*n_importance)

  # Sample from cdf and the corresponding bin centers.
  cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, n_importance, 2)
  bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, n_importance, 2)

  # Convert samples to ray length.
  denom = (cdf_g[..., 1] - cdf_g[..., 0])
  denom[denom<1e-5] = 1
  samples = bins_g[..., 0] + (u-cdf_g[...,0])/denom * (bins_g[..., 1] - bins_g[..., 0])

  return samples # [n_rays, n_importance]

def sample_hierarchical(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  z_vals: torch.Tensor,
  weights: torch.Tensor,
  n_importance: int,
  perturb: bool = False
):
  """
  Apply hierarchical sampling to rays.
  Return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  """

  # Sample PDF using z_vals as bins and weights as probabilities.
  z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1]) # (N_rays, n_importance-1)
  new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_importance,
                          perturb=perturb)
  new_z_samples = new_z_samples.detach()

  # Resample points from ray based on PDF.
  z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
  pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_importance, 3]
  return pts, z_vals_combined, new_z_samples
