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