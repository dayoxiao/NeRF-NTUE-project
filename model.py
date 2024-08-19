import torch
from torch import nn

class PositionalEncoder(nn.Module):
  """
  Sin cos positional encoder
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_sampling: bool = True
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_sampling = log_sampling
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Frequencies in linear or log scale
    if self.log_sampling:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

  def forward(
    self,
    x
  ):
    """
    Encode input
    Out: torch.tensor
    """
    #[80000, 63 = 3 + 3x10x2 = 63]
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class NeRF(nn.Module):
  """
  NeRF module.
  """
  def __init__(
    self,
    d_input: int = 3,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: int = (4,),
    d_viewdirs: int = None
  ):
    super().__init__()
    self.d_input = d_input
    self.skip = skip
    self.d_viewdirs = d_viewdirs

    # Model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter, d_filter) if i not in skip
        else nn.Linear(d_filter + self.d_input, d_filter) for i in range(n_layers - 1)]
    )

    # Output layers
    if self.d_viewdirs is not None:
      # If viewdirs, split sigma and RGB
      # Sigma
      self.sigma_out = nn.Linear(d_filter, 1)
      # RGB
      self.feature_filters = nn.Linear(d_filter, d_filter)
      self.views_linears = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # Simple output
      self.output = nn.Linear(d_filter, 4)

  def forward(
    self,
    x: torch.Tensor,
    viewdirs: torch.Tensor = None,
    sigma_only=False
  ):
    """
    Forward passing with optional view direction.
    sigma_only: infer only sigma value.
    Return: torch.tensor
    """
    # d_viewdirs error check
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('d_viewdirs was not given.')

    # Model layers forward passing
    x_input = x
    for i, layer in enumerate(self.layers):
      x = nn.functional.relu(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Output layer
    # Sigma
    sigma = self.sigma_out(x)
    if sigma_only:
      return sigma
    
    if self.d_viewdirs is not None:
      # RGB
      rgb_feature = self.feature_filters(x)
      cat = torch.concat([rgb_feature, viewdirs], dim=-1)
      cat = nn.functional.relu(self.views_linears(cat))
      rgb = self.output(cat)

      # Concat output
      out = torch.concat([rgb, sigma], dim=-1)
    else:
      # Simple output
      out = self.output(x)
    return out