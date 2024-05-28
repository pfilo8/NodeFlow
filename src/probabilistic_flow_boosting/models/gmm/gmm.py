import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np


class GaussianMixture(torch.nn.Module):
    def __init__(self, n_components, context_dim, output_dim = 1, epsilon=1.e-6):
        super(GaussianMixture, self).__init__()
        self.n_components = n_components
        self.context_dim = context_dim
        self.epsilon = epsilon
        self.output_dim = output_dim

        self.forward_layer = nn.Linear(self.context_dim, self.n_components * 3)
        # Initialize mixture coefficient logits to near zero so that mixture coefficients
        # are approximately uniform.
        self.forward_layer.weight.data[::3, :] = self.epsilon * torch.randn(
            self.n_components, self.context_dim
        )
        self.forward_layer.bias.data[::3] = self.epsilon * torch.randn(
            self.n_components
        )
        # Initialize unconstrained standard deviations to the inverse of the softplus
        # at 1 so that they're near 1 at initialization.
        self.forward_layer.weight.data[2::3] = self.epsilon * torch.randn(
            self.n_components, self.context_dim
        )
        self.forward_layer.bias.data[2::3] = torch.log(torch.exp(torch.Tensor([1 - self.epsilon])) - 1) * \
            torch.ones(self.n_components) + self.epsilon * torch.randn(self.n_components)


    def log_prob(self, context, y):
        outputs = self.forward_layer(context)
        outputs = outputs.reshape(*y.shape, self.n_components, 3)

        logits, means, unconstrained_stds = (
            outputs[..., 0],
            outputs[..., 1],
            outputs[..., 2],
        )

        log_mixture_coefficients = torch.log_softmax(logits, dim=-1)
        stds = F.softplus(unconstrained_stds) + self.epsilon

        log_prob = torch.sum(
            torch.logsumexp(
                log_mixture_coefficients
                - 0.5
                * (
                    np.log(2 * np.pi)
                    + 2 * torch.log(stds)
                    + ((y[..., None] - means) / stds) ** 2
                ),
                dim=-1,
            ),
            dim=-1,
        )
        return log_prob
    
    def sample(self, context, num_samples=1000):
        context = context.repeat(num_samples, 1)
        with torch.no_grad():
            samples = torch.zeros(context.shape[0], self.output_dim, device=context.device)
            for feature in range(self.output_dim):
                outputs = self.forward_layer(context)
                outputs = outputs.reshape(*samples.shape, self.n_components, 3)
                logits, means, unconstrained_stds = (
                    outputs[:, feature, :, 0],
                    outputs[:, feature, :, 1],
                    outputs[:, feature, :, 2],
                )
                logits = torch.log_softmax(logits, dim=-1)
                stds = F.softplus(unconstrained_stds) + self.epsilon
                component_distribution = dist.Categorical(logits=logits)
                components = component_distribution.sample((1,)).reshape(-1, 1)
                means, stds = (
                    means.gather(1, components).reshape(-1),
                    stds.gather(1, components).reshape(-1),
                )
                samples[:, feature] = (
                    means + torch.randn(context.shape[0], device=samples.device) * stds
                ).detach()
        return samples.reshape(-1, num_samples, self.output_dim)