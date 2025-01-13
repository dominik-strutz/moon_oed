import torch
from torch import nn
from zuko.nn import MLP

class ConditionalGaussian(nn.Module):
    def __init__(
        self,
        N_rec,
        features,
        design_mean=None, design_std=None,
        data_mean=None, data_std=None,
        model_mean=None, model_std=None,    
        **kwargs
    ):
        super(ConditionalGaussian, self).__init__()

        context = ((N_rec -1) * N_rec) // 2 
   
        self.hyper = MLP(
            in_features  = N_rec*2 + context,
            out_features = features*2,
            **kwargs
        )
        
        # register the hyper as a submodule
        self.add_module('hyper', self.hyper)
        
        # Store the normalization parameters
        self.design_mean = design_mean
        self.design_std = design_std
        self.data_mean = data_mean
        self.data_std = data_std
        self.model_mean = model_mean
        self.model_std = model_std
        
    def forward(self, data, design):   
        # Normalize the design and data
        if (self.design_mean is not None) and (self.design_std is not None):
            design = (design - self.design_mean) / self.design_std
        
        if (self.data_mean is not None) and (self.data_std is not None):
            data = (data - self.data_mean) / self.data_std
        
        design = design.flatten(start_dim=-2)
        x = torch.cat([data, design], dim=-1)
        
        x = self.hyper(x)
        mu, log_sigma = x.chunk(2, dim=-1)
        
        if (self.model_mean is not None) and (self.model_std is not None):
            mu = (mu * self.model_std) + self.model_mean
            log_sigma = log_sigma + self.model_std.log()
        
        return torch.distributions.Independent(
            torch.distributions.Normal(
                mu, log_sigma.exp()), 1
        )
        
        # # Add a small value to upper bound since the truncated distribution is exclusive at the upper bound
        
        # # coordinate_dist = Independent(Truncated(
        #     # Normal(mu[:, :2], log_sigma[:, :2].exp()),
        #     # lower=0.0, upper=length_x+1e-6), 1)
        # # std_dist = Independent(Truncated(
        #     # # Normal(mu[:, 2:4], log_sigma[:, 2:4].exp()),
        #     # lower=10.0, upper=100.0+1e-6), 1)
        # x_dist     = Independent(Truncated(Normal(mu[:, [0]], log_sigma[:, [0]].exp()), lower=0.0,  upper=length_x+1e-6), 1)
        # y_dist     = Independent(Truncated(Normal(mu[:, [1]], log_sigma[:, [1]].exp()), lower=0.0,  upper=length_y+1e-6), 1)
        # std_x_dist = Independent(Truncated(Normal(mu[:, [2]], log_sigma[:, [2]].exp()), lower=10.0, upper=100.0+1e-6), 1)
        # std_y_dist = Independent(Truncated(Normal(mu[:, [3]], log_sigma[:, [3]].exp()), lower=10.0, upper=100.0+1e-6), 1)
        # amp_dist   = Independent(Normal(mu[:, [4]], log_sigma[:, [4]].exp()), 1)
        # theta_dist = Independent(Truncated(Normal(mu[:, [5]], log_sigma[:, [5]].exp()), lower=0.0, upper=0.5*torch.pi+1e-6), 1)
        
        # return Joint(x_dist, y_dist, std_x_dist, std_y_dist, amp_dist, theta_dist)
        
    def to(self, *args, **kwargs):
        self.design_mean = self.design_mean.to(*args, **kwargs) if self.design_mean is not None else None
        self.design_std = self.design_std.to(*args, **kwargs) if self.design_std is not None else None
        self.data_mean = self.data_mean.to(*args, **kwargs) if self.data_mean is not None else None
        self.data_std = self.data_std.to(*args, **kwargs) if self.data_std is not None else None
        self.model_mean = self.model_mean.to(*args, **kwargs) if self.model_mean is not None else None
        self.model_std = self.model_std.to(*args, **kwargs) if self.model_std is not None else None
        return super(ConditionalGaussian, self).to(*args, **kwargs)
    
    def cpu(self):
        self.design_mean = self.design_mean.cpu() if self.design_mean is not None else None
        self.design_std = self.design_std.cpu() if self.design_std is not None else None
        self.data_mean = self.data_mean.cpu() if self.data_mean is not None else None
        self.data_std = self.data_std.cpu() if self.data_std is not None else None
        self.model_mean = self.model_mean.cpu() if self.model_mean is not None else None
        self.model_std = self.model_std.cpu() if self.model_std is not None else None
        return super(ConditionalGaussian, self).cpu()