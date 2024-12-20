import torch
from torch.distributions import Normal, Independent

def erf_chebyshev_approximation(x):
    """
    Fast Chebyshev approximation of the error function using PyTorch.
    """
    # Define constants
    c6, c5, c4, c3, c2, c1, c0 = 0.0145688, -0.0348595, 0.0503913, -0.0897001, 0.156097, -0.249431, 0.533201
    z_factor = 0.289226
    # Condition for splitting
    threshold = 2.629639
    # Compute z for |x| <= threshold
    z = z_factor * x**2 - 1
    # Piecewise function
    erf_approx = torch.where(
        torch.abs(x) > threshold,
        torch.sign(x),  # sgn(x) when |x| > threshold
        (c6 * z**6 + c5 * z**5 + c4 * z**4 + c3 * z**3 + c2 * z**2 + c1 * z + c0) * x  # Polynomial approx for |x| <= threshold
    )

    return erf_approx

def integrate_multiple_gaussians(
    ray_start, ray_end,
    gaussian_mean,
    gaussian_std,
    gaussian_amp,
    gaussian_theta):
    """
    Integrates multiple anisotropic Gaussians over multiple rays.

    Parameters:
        ray_start (Tensor): Starting points of the rays, shape (..., N_rays, 2/3)
        ray_end (Tensor): Ending points of the rays, shape (..., N_rays, 2/3)
        gaussian_means (Tensor): Centers of the Gaussians, shape (..., K, 2/3)
        gaussian_std (Tensor): Standard deviation parameters (a_x, a_y, a_z) of the Gaussians, shape (..., K, 3)
        gaussian_amps (Tensor): Amplitudes of the Gaussians, shape (..., K)
        gaussian_theta (Tensor): Rotation angles around the z-axis, shape (..., K)

    Returns:
        Tensor: Sum of the Gaussian integrals for each ray, shape (..., N_rays, 1).
    """
    N_rays = ray_start.shape[-2]
    K = gaussian_mean.shape[-2]
    N_dim = ray_start.shape[-1]
    
    if N_dim not in [2, 3]:
        raise ValueError("Only 2D and 3D rays are supported.")

    # Convert from std to scale
    gaussian_scale = 1/gaussian_std
    
    # add dimension for broadcasting
    if gaussian_amp.dim() == 1:
        gaussian_amp = gaussian_amp.unsqueeze(-1)  # (..., K, 1)
    if gaussian_theta.dim() == 1:
        gaussian_theta = gaussian_theta.unsqueeze(-1)

    # Expand dimensions for broadcasting
    ray_start_exp = ray_start.unsqueeze(-2)  # (..., N_rays, 1, N_dim)
    ray_end_exp = ray_end.unsqueeze(-2)  # (..., N_rays, 1, N_dim)

    gaussian_mean_exp = gaussian_mean.unsqueeze(-3)  # (..., 1, K, N_dim)
    gaussian_scale_exp = gaussian_scale.unsqueeze(-3)  # (..., 1, K, N_dim)
    gaussian_amp_exp = gaussian_amp.unsqueeze(-3)  # (..., 1, K, 1)
    gaussian_theta_exp = gaussian_theta.unsqueeze(-3)  # (..., 1, K)

    # Compute rotation matrix for each Gaussian
    cos_theta = torch.cos(gaussian_theta_exp)  # (..., 1, K)
    sin_theta = torch.sin(gaussian_theta_exp)  # (..., 1, K)
    if N_dim == 2:
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2).reshape(1, K, 2, 2) # (..., 1, K, N_dim, N_dim)
    else:
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(sin_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.zeros_like(sin_theta), torch.ones_like(cos_theta)], dim=-1)
        ], dim=-2).reshape(1, K, 3, 3) # (..., 1,K, N_dim, N_dim)

    # Rotate ray_start and ray_end around the Gaussian means
    ray_start_rot = torch.bmm(
        (ray_start_exp - gaussian_mean_exp).unsqueeze(-2).flatten(end_dim=-3),
        rotation_matrix.repeat(ray_start_exp.shape[0], 1, 1, 1).flatten(end_dim=-3)  # (..., N_rays, K, 3, 3)
        ).reshape(N_rays, K, N_dim) + gaussian_mean_exp  # (..., N_rays, K, N_dim)
    ray_end_rot = torch.bmm(
        (ray_end_exp - gaussian_mean_exp).unsqueeze(-2).flatten(end_dim=-3),
        rotation_matrix.repeat(ray_end_exp.shape[0], 1, 1, 1).flatten(end_dim=-3)  # (..., N_rays, K, 3, 3)
        ).reshape(N_rays, K, N_dim) + gaussian_mean_exp  # (..., N_rays, K, N_dim)

    # Compute new ray direction and length after rotation
    ray_direction_rot = ray_end_rot - ray_start_rot  # (..., N_rays, K, N_dim)
    ray_lengths_rot = torch.norm(ray_direction_rot, dim=-1, keepdim=True)  # (..., N_rays, K, 1)
    ray_directions_norm_rot = ray_direction_rot / ray_lengths_rot  # (..., N_rays, K, N_dim)

    # Compute b' = b - v and a_i^2
    b_prime = gaussian_mean_exp - ray_start_rot  # (..., N_rays, K, N_dim)
    a_squared = gaussian_scale_exp**2  # (..., N_rays, K, N_dim)

    # Compute A, B, C
    A = torch.sum(a_squared * ray_directions_norm_rot**2, dim=-1, keepdim=True)  # (..., N_rays, K, 1)
    B = 2 * torch.sum(a_squared * ray_directions_norm_rot * b_prime, dim=-1, keepdim=True)  # (..., N_rays, K, 1)
    C = torch.sum(a_squared * b_prime**2, dim=-1, keepdim=True)  # (..., N_rays, K, 1)

    # Prefactor
    prefactor = gaussian_amp_exp * torch.sqrt(torch.tensor(torch.pi)) / (2 * torch.sqrt(A))  # (..., N_rays, K, 1)
    exp_term = torch.exp(-C + B**2 / (4 * A))  # (..., N_rays, K, 1)

    # Error function arguments
    u0 = -B / (2 * torch.sqrt(A))  # (..., N_rays, K, 1)
    u1 = torch.sqrt(A) * ray_lengths_rot + u0  # (..., N_rays, K, 1)

    # Compute erf difference and final integral
    erf_diff = erf_chebyshev_approximation(u1) - erf_chebyshev_approximation(u0)  # (..., N_rays, K, 1)
    integrals = prefactor * exp_term * erf_diff  # (..., N_rays, K, 1)

    # Sum over the Gaussians for each ray
    result = torch.sum(integrals/ray_lengths_rot, dim=-2).squeeze(-1)  # (..., N_rays)

    return result

class ForwardModelHomogeneous:
    def __init__(self, background_velocity):
        self.background_velocity = background_velocity
        
    def __call__(
        self,
        model_parameters,
        design):

        combinations = torch.combinations(
            torch.tensor(range(len(design))),
            2, with_replacement=False)

        start_points = design[combinations[:, 0]]
        end_points = design[combinations[:, 1]]
        distances = torch.norm(start_points - end_points, dim=1)
        
        # calculate the distance between the receivers
        
        integrals = integrate_multiple_gaussians(
            start_points, end_points,
            gaussian_mean=model_parameters[..., :2],
            gaussian_std=model_parameters[..., 2:4],
            gaussian_amp=model_parameters[..., 4],
            gaussian_theta=model_parameters[..., 5],
        )

        phase_arrivals = (distances) / self.background_velocity  + integrals  / self.background_velocity
        
        return phase_arrivals
    
class DataLikelihoodHomogeneous:
    def __init__(self, fwd_function, noise_std):
        self.fwd_function = fwd_function
        self.noise_std = noise_std
    def __call__(
        self,
        model_parameters,
        design,):
        
        if model_parameters.dim() == 1:
            model_parameters = model_parameters.unsqueeze(0)
        
        fwd = self.fwd_function(model_parameters, design)
        
        return Independent(Normal(fwd, self.noise_std),1)