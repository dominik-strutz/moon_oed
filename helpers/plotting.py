from matplotlib.patches import Ellipse
import numpy as np
import torch

def plot_ellipses(
    ax, gaussian_means, gaussian_std, gaussian_theta, gaussian_amplitudes,
    scatter_kwargs={}, ellipse_kwargs={},
    n_std=2, n_max=1000, ):
    
    scatter_kwargs.setdefault('s', 50)
    scatter_kwargs.setdefault('c', 'tab:blue')
    scatter_kwargs.setdefault('linewidths', 0.0)
    scatter_kwargs.setdefault('marker', 'o')
    
    ellipse_kwargs.setdefault('edgecolor', 'tab:blue')
    ellipse_kwargs.setdefault('linewidth', 2.0)
    ellipse_kwargs.setdefault('facecolor', 'none')
    
    for mean, std, theta, amp in zip(gaussian_means[:n_max], gaussian_std[:n_max], gaussian_theta[:n_max], gaussian_amplitudes[:n_max]):
        ax.scatter(
            mean[0], mean[1],
            alpha=amp.item() / gaussian_amplitudes.max().item(),
            **scatter_kwargs)
        
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=n_std*std[0], height=n_std*std[1],
            angle=-theta * 180 / np.pi,
            alpha=amp.item() / gaussian_amplitudes.max().item(),
            **ellipse_kwargs)
        
        ax.add_patch(ellipse)  # Plot the 1st standard deviation ellipse
    
    
def gaussian_2d(x, y, mean, std, amplitude, theta):
    x0 = mean[..., 0]
    y0 = mean[..., 1]
    
    sigma_x = std[..., 0]
    sigma_y = std[..., 1]
    
    a = torch.cos(theta)**2 / (2 * sigma_x**2) + torch.sin(theta)**2 / (2 * sigma_y**2)
    b = -torch.sin(2 * theta) / (4 * sigma_x**2) + torch.sin(2 * theta) / (4 * sigma_y**2)
    c = torch.sin(theta)**2 / (2 * sigma_x**2) + torch.cos(theta)**2 / (2 * sigma_y**2)
    
    return amplitude * torch.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))
