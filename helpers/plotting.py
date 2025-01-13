import matplotlib.pyplot as plt
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
        
        scatter_kwargs.setdefault('alpha', amp.item() / gaussian_amplitudes.max().item())
        ellipse_kwargs.setdefault('alpha', amp.item() / gaussian_amplitudes.max().item())
        
        ax.scatter(
            mean[0], mean[1],
            **scatter_kwargs)
        if 'label' in scatter_kwargs:
            del scatter_kwargs['label']
        
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=n_std*std[0], height=n_std*std[1],
            angle=-theta * 180 / np.pi,
            **ellipse_kwargs)
        ax.add_patch(ellipse)  # Plot the 2nd standard deviation ellipse
        if 'label' in ellipse_kwargs:
            del ellipse_kwargs['label']

        
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


def plot_marginal_histogramms(
    prior_samples, posterior_samples,
    param_limits=None, true_model=None, parameter_names=None,
    show=True
    ):
    
    fig, ax = plt.subplots(1, prior_samples.shape[-1], figsize=(2*prior_samples.shape[-1], 2))

    for i, ax_i in enumerate(ax):
        ax_i.hist(
            prior_samples[..., i].flatten(), bins=50, alpha=0.5,
            color='gray', density=True,
            range=param_limits[i] if param_limits is not None else None
            )
        ax_i.hist(
            posterior_samples[:, i], bins=50, color='tab:red',
            density=True,
            range=param_limits[i] if param_limits is not None else None
            )
        
        if parameter_names is not None:
            ax_i.set_title(parameter_names[i]) 

        if true_model is not None:
            ax_i.axvline(true_model[i].item(), color='k')

    if show:
        plt.tight_layout()
        plt.show()
    else:
        return fig, ax
    
    
def plot_single_posterior(
    posterior_samples,
    true_model,
    design, length_x, length_y):
    
    if true_model.dim() == 1:
        true_model = true_model.unsqueeze(0)
    
    x = torch.linspace(0, length_x, 100)
    y = torch.linspace(0, length_y, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()

    few_samples = posterior_samples[:30]

    post_means = few_samples[..., :2]
    post_stds = few_samples[..., 2:4]
    post_amps = few_samples[..., 4]
    post_thetas = few_samples[..., 5]

    average_posterior = gaussian_2d(
        X.unsqueeze(-1), Y.unsqueeze(-1),
        post_means,
        post_stds,
        post_amps,
        post_thetas).reshape(len(x), len(y), len(few_samples)).mean(dim=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150, sharex=True, sharey=True)

    plot_ellipses(
        ax1, post_means, post_stds, post_thetas, post_amps,
        scatter_kwargs={'c': 'tab:red', 's':2, 'alpha':0.4},
        ellipse_kwargs={'edgecolor': 'tab:red', 'linewidth': 0.5, 'label': 'posterior samples', 'alpha':0.4},
        n_max=100)

    ax1.set_title('posterior ice bodies')
    
    im = ax2.imshow(
        average_posterior.T, extent=(0, length_x, 0, length_y),
        origin='lower', cmap='seismic', zorder=-1, clim=(-abs(average_posterior).max(), abs(average_posterior).max()))
    ax2.set_title('mean posterior wave speed')

    # add axes below ax2 for colorbar
    cbarax = fig.add_axes([0.6, 0.02, 0.25, 0.02])

    cbar = fig.colorbar(im, cax=cbarax, orientation='horizontal', shrink=0.6, label='mean wave speed difference')

    for ax in (ax1, ax2):
                
        plot_ellipses(
            ax, true_model[..., :2], true_model[...,2:4], true_model[...,5], true_model[..., 4],
            scatter_kwargs={'c': 'k', 's':10},
            ellipse_kwargs={'edgecolor': 'k', 'linewidth': 2.5, 'label': 'true model'})
        buffer=50

        ax.set_xlim(0-buffer, length_x+buffer)
        ax.set_ylim(0-buffer, length_y+buffer)

        # draw rectangle
        ax.add_patch(plt.Rectangle(
            (0, 0), length_x, length_y,
            fill=None, linestyle='dashed', color='black', linewidth=2))
        ax.set_aspect('equal')
            
        ax.scatter(design[:, 0], design[:, 1],
                    linewidths=0,
                    marker='^', color='k', s=80)

    # remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # legend below plot
    ax1.legend(by_label.values(), by_label.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  ncol=1, fontsize='small')

    plt.show()
    
    
def plot_n_posterior(
    prior_dist,
    forward_model,
    conditional_gaussian,
    design,
    length_x,
    length_y,
    plot_mean=True,
    plot_samples=False,
):
    
    # Sample 10 true solutions from the prior
    torch.manual_seed(0)
    true_solutions = prior_dist.sample((10,))

    init_models = prior_dist.sample((1000,))
    std_init_models = init_models.std(dim=0, keepdim=True)

    # Generate observed data
    observed_data = forward_model(true_solutions, design)

    # Generate posterior samples using the ConditionalADVI model
    posterior_means = []
    posterior_stds = []
    for i in range(10):
        posterior = conditional_gaussian(observed_data[i].unsqueeze(0), design.unsqueeze(0))    
        posterior_means.append(posterior.sample((1000,)).mean(dim=0))
        # append average normalized std
        posterior_stds.append((posterior.sample((1000,)).std(dim=0) / std_init_models).mean(dim=-1))
    
    
    # Plot the true solutions and posterior means in a 5x2 grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), dpi=200)
    fig.set_facecolor('white')

    for i, ax in enumerate(axes.flatten()):    
        
        if plot_mean:
            pm = posterior_means[i][0]

            plot_ellipses(ax, pm[:2].unsqueeze(0), pm[2:4].unsqueeze(0),
                        pm[5].unsqueeze(0), pm[4].unsqueeze(0),
                        scatter_kwargs={'c': 'tab:red', 's': 20, 'label': 'posterior mean (center)'},
                        ellipse_kwargs={'edgecolor': 'tab:red', 'linewidth': 2.0, 'label': 'posterior mean (size)'})
        if plot_samples:
            posterior = conditional_gaussian(observed_data[i].unsqueeze(0), design.unsqueeze(0))
            posterior_samples = posterior.sample((30,)).squeeze(1)

            sample_means = posterior_samples[..., :2]
            sample_stds = posterior_samples[..., 2:4]
            sample_amps = posterior_samples[..., 4]
            sample_thetas = posterior_samples[..., 5]    

            plot_ellipses(
                ax, sample_means, sample_stds, sample_thetas, sample_amps,
                scatter_kwargs={'c': 'tab:red', 's': 5,
                                'label': 'posterior sample (center)'},
                ellipse_kwargs={'edgecolor': 'tab:red', 'linewidth': 0.5, 
                                'label': 'posterior sample (size)'},
            )

        plot_ellipses(ax, true_solutions[i, :2].unsqueeze(0), true_solutions[i, 2:4].unsqueeze(0),
                      true_solutions[i, 5].unsqueeze(0), true_solutions[i, 4].unsqueeze(0),
                      scatter_kwargs={'c': 'tab:blue', 's': 20, 'label': 'true model (center)'},
                      ellipse_kwargs={'edgecolor': 'tab:blue', 'linewidth': 2.0, 'label': 'true  (size)'})

        ax.scatter(
            design[:, 0], design[:, 1], marker='^', color='k', s=40,
            linewidth=0.0, label='receivers')

        buffer = 50
        ax.set_xlim(0-buffer, length_x+buffer)
        ax.set_ylim(0-buffer, length_y+buffer)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.add_patch(plt.Rectangle((0, 0), length_x, length_y, fill=None, linestyle='dashed', color='black', linewidth=1))
        ax.set_aspect('equal')

        # add average std of the posterior
        ax.title.set_text(fr' $\sigma_\text{{post}} / \sigma_\text{{prior}}$: {posterior_stds[i].mean().item():.2f}')

    # add legend below the plots at half the width of the figure
    # remove duplicate legend entries
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # legend below plot
    fig.legend(by_label.values(), by_label.keys(),
        loc='upper center', ncol=3, 
        bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.show()