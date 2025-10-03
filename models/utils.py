import torch
import torch.nn.functional as F


def gumbel_softmax_binary(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax for binary values
    
    This allows differentiable sampling from a Bernoulli distribution,
    which is crucial for training the cVAE with binary outputs.
    
    Args:
        logits: [B, 1, H, W] raw outputs from decoder
        temperature: float, controls discretization (lower = more discrete)
        hard: if True, returns hard {0, 1}, else soft probabilities
    
    Returns:
        [B, 1, H, W] differentiable binary-like values
    """
    # Convert to binary logits: [logit, -logit] for [class_1, class_0]
    logits_binary = torch.stack([logits, -logits], dim=-1)  # [B, 1, H, W, 2]
    
    # Add Gumbel noise
    gumbels = -torch.log(-torch.log(torch.rand_like(logits_binary) + 1e-20) + 1e-20)
    gumbels = (logits_binary + gumbels) / temperature
    
    # Softmax
    y_soft = F.softmax(gumbels, dim=-1)[..., 0]  # Take first class [B, 1, H, W]
    
    if hard:
        # Straight-through estimator: forward pass uses hard values,
        # backward pass uses soft gradients
        y_hard = (y_soft > 0.5).float()
        y = y_hard - y_soft.detach() + y_soft  # Gradient flows through y_soft
    else:
        y = y_soft
    
    return y


def sample_gumbel(shape, device='cuda', eps=1e-20):
    """Sample from Gumbel(0, 1) distribution"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def temperature_schedule(epoch, total_epochs, start_temp=5.0, end_temp=0.5):
    """
    Linear temperature annealing schedule for Gumbel-Softmax
    
    Args:
        epoch: current epoch
        total_epochs: total number of epochs
        start_temp: initial temperature (high = soft)
        end_temp: final temperature (low = hard)
    
    Returns:
        current temperature
    """
    return start_temp + (end_temp - start_temp) * (epoch / total_epochs)
