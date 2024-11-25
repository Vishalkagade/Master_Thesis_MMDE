import torch
import os
import numpy as np
from tqdm import tqdm
import wandb



# EarlyStopping class (as you've defined it)
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

# Checkpoint class (as you've defined it)
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

        depth_state_dict = model.state_dict()

        if self.save_model_only:
            checkpoint = depth_state_dict
            file_name += ".pt"
        else:
            checkpoint = {
                'depth_head': depth_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }       
            file_name += ".tar"

        torch.save(checkpoint, os.path.join(self.ckpt_dir, file_name))


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, fabric, epoch, num_steps, mode="train", eval_step=5, num_epochs=100):
    is_training = (mode == "train")
    if is_training:
      model.train()
    else:
      model.eval()
    prefix = "train" if is_training else "val"

    running_loss = 0.0
    accumulated_metrics = {key: 0.0 for key in ["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "mae", "d1", "d2", "d3"]}
    pbar = tqdm(enumerate(dataloader, start=1), total=num_steps, leave=True)

    for steps, (image, mmde_map, radar_img, ground_truth) in pbar:
        if is_training:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(image, radar_img, mmde_map)
            mask = outputs > 0.01
            loss = criterion(outputs, ground_truth,mask)

            fabric.clip_gradients(model, optimizer, max_norm=2.0, error_if_nonfinite=False)
            # Backward pass and optimization (only if training)
            fabric.backward(loss)
            optimizer.step()

        else:  # Validation mode
            with torch.no_grad():
                outputs = model(image, radar_img, mmde_map)
                mask = outputs > 0.01
                valid_pixels = torch.sum(mask).item()
                #print(f"Number of valid pixels in mask: {valid_pixels}")
                loss = criterion(outputs, ground_truth,mask)

        running_loss += loss.item()
    if is_training:
      scheduler.step()
    final_loss = running_loss / num_steps


    pbar.set_description(f"Epoch: [{epoch}/{num_epochs}]")
    pbar.set_postfix({f"{prefix}_loss": final_loss, "lr": scheduler.get_last_lr()[0]})

    wandb.log({f"{prefix}_epoch_loss": final_loss, "epoch": epoch})

    return final_loss