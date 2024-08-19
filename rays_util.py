import torch

def get_rays(
  height: int,
  width: int,
  focal_length: float,
  c2w: torch.Tensor
):
  """
  Shot rays through every picture pixel and find camera origin.
  Return: Tuple[torch.Tensor, torch.Tensor]
  """

  # Apply pinhole camera model to gather directions at each pixel
  i, j = torch.meshgrid(
      torch.arange(width, dtype=torch.float32).to(c2w),
      torch.arange(height, dtype=torch.float32).to(c2w), indexing="xy")
  directions = torch.stack([(i - width * .5) / focal_length,
                            -(j - height * .5) / focal_length,
                            -torch.ones_like(i)
                           ], dim=-1)

  # Apply camera pose to change directions from camera to world coordination
  rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

  # Origin
  rays_o = c2w[:3, -1].expand(rays_d.shape)
  return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # projection
    o0 = -1. / (W / (2. * focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1. / (H / (2. * focal)) * rays_o[...,1] / rays_o[...,2]
    o2 =  1. +  2. * near / rays_o[...,2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d