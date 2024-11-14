import torch
import numpy as np
import os

# callbacks
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_current_checkpoint = True
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_current_checkpoint = False
            if self.verbose:
                print(f'[EarlyStopping] Counter: {self.counter}/ {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            self.save_current_checkpoint = True

class Checkpoint:
    def __init__(self, root_dir, experiment_name, checkpoint_name=None, save_model_only=False):
        self.ckpt_dir = os.path.join(root_dir, experiment_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.checkpoint_name = experiment_name if checkpoint_name is None else checkpoint_name
        self.save_model_only = save_model_only
            
    def __call__(self, epoch, t_loss, v_loss, model, optimizer, scheduler):

        file_name = f"{self.checkpoint_name}_E{epoch}_L{t_loss:.4f}_VL{v_loss:.4f}"
        
        if hasattr(model, "_original_module"):
            # fabric'ed model
            model = model._original_module

        detect_head_state_dict = model.detect_head.state_dict()

        if self.save_model_only:
            checkpoint = detect_head_state_dict
            file_name += ".pth"
        else:
            checkpoint = {
                'detect_head': detect_head_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }       
            file_name += ".tar"

        torch.save(checkpoint, os.path.join(self.ckpt_dir, file_name))