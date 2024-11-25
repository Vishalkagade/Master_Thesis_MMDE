import numpy as np
import torch
from prettytable import PrettyTable

def compute_errors(gt, pred):
    # Convert inputs to PyTorch tensors
    gt = torch.tensor(gt, dtype=torch.float32) if not isinstance(gt, torch.Tensor) else gt
    pred = torch.tensor(pred, dtype=torch.float32) if not isinstance(pred, torch.Tensor) else pred
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    gt = torch.clamp(gt, min=epsilon)
    pred = torch.clamp(pred, min=epsilon)
    
    # Threshold-based accuracy metrics
    thresh = torch.max(gt / pred, pred / gt)
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    # Error metrics
    mae = torch.mean(torch.abs(gt - pred))  # Mean Absolute Error
    rms = torch.sqrt(torch.mean((gt - pred) ** 2))  # Root Mean Squared Error
    log_rms = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))  # Log Root Mean Squared Error
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)  # Absolute Relative Error
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)  # Squared Relative Error

    # Scale-Invariant Logarithmic Error (SiLog)
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    # Log10 Error
    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.mean(err)

    # Return all metrics as a list
    return {
        "silog": silog.item(),
        "abs_rel": abs_rel.item(),
        "log10": log10.item(),
        "rms": rms.item(),
        "sq_rel": sq_rel.item(),
        "log_rms": log_rms.item(),
        "mae": mae.item(),
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
    }

from collections import defaultdict

import torch

def compute_errors_batches(gt, pred):
    # Ensure gt and pred are PyTorch tensors
    device = gt.device if isinstance(gt, torch.Tensor) else 'cpu'
    gt = torch.tensor(gt, dtype=torch.float32, device=device) if not isinstance(gt, torch.Tensor) else gt
    pred = torch.tensor(pred, dtype=torch.float32, device=device) if not isinstance(pred, torch.Tensor) else pred

    # Check batch dimensions and reshape if needed
    if len(gt.shape) > 2:  # Assuming input shape is [B, C, H, W]
        gt = gt.squeeze(1)  # Remove channel dimension, now [B, H, W]
        pred = pred.squeeze(1)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    gt = torch.clamp(gt, min=epsilon)
    pred = torch.clamp(pred, min=epsilon)

    # Threshold-based accuracy metrics
    thresh = torch.max(gt / pred, pred / gt)
    d1 = (thresh < 1.25).float().mean(dim=[1, 2])
    d2 = (thresh < 1.25 ** 2).float().mean(dim=[1, 2])
    d3 = (thresh < 1.25 ** 3).float().mean(dim=[1, 2])

    # Error metrics (batch-wise computation)
    mae = torch.mean(torch.abs(gt - pred), dim=[1, 2])  # Mean Absolute Error
    rms = torch.sqrt(torch.mean((gt - pred) ** 2, dim=[1, 2]))  # Root Mean Squared Error
    log_rms = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2, dim=[1, 2]))  # Log RMS Error
    abs_rel = torch.mean(torch.abs(gt - pred) / gt, dim=[1, 2])  # Absolute Relative Error
    sq_rel = torch.mean(((gt - pred) ** 2) / gt, dim=[1, 2])  # Squared Relative Error

    # Scale-Invariant Logarithmic Error (SiLog)
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2, dim=[1, 2]) - (torch.mean(err, dim=[1, 2]) ** 2)) * 100

    # Log10 Error
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(gt)), dim=[1, 2])

    # Averaging metrics over the batch
    metrics = {
        "silog": silog.mean().item(),
        "abs_rel": abs_rel.mean().item(),
        "log10": log10.mean().item(),
        "rms": rms.mean().item(),
        "sq_rel": sq_rel.mean().item(),
        "log_rms": log_rms.mean().item(),
        "mae": mae.mean().item(),
        "d1": d1.mean().item(),
        "d2": d2.mean().item(),
        "d3": d3.mean().item(),
    }

    return metrics

def compute_metrics(ground_truth, depth_map, accumulated_metrics):
    batch_metrics = compute_errors_batches(ground_truth, depth_map)
    for key in accumulated_metrics.keys():
        accumulated_metrics[key] += batch_metrics[key]

# Function to display metrics using PrettyTable
def display_metrics(accumulated_metrics, num_steps):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    # Average the metrics and add them to the table
    for metric, value in accumulated_metrics.items():
        avg_value = value / num_steps
        table.add_row([metric, f"{avg_value:.4f}"])
        accumulated_metrics[metric] = avg_value  # Update the dict with averaged values

    print(table)