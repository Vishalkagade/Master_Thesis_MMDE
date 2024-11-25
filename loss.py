import torch
import torch.nn as nn

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.l1_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
    

class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.mse_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
    

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
    

