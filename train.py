import os
os.chdir('Master_Thesis_MMDE_latest')
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from lightning.fabric import Fabric
import wandb
from tqdm.notebook import tqdm
from prettytable import PrettyTable
from loss import l1_loss
from train_helper import EarlyStopping, Checkpoint,train_one_epoch

from dataset import TrainDataLoader,ValDataLoader

from TransMIRNet import fusion_net

from sklearn.model_selection import train_test_split

def read_image(path):
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32)/256
    return input_sparse_depth

class Params:
    def __init__(self):
        self.radar_input_channels = 1
        self.encoder_radar = 'resnet18'
        self.mmde_encoder = "resnet18"
        self.activation = 'elu' # why?
        self.encoder = 'resnet34'
        self.max_depth = 4


def main():
    IMAGE_HEIGHT = 320
    IMAGE_WIDTH = 1280
    BATCH_SIZE = 1
    CROP_SCALE_HEIGHT = 0.4
    CROP_SCALE_WIDTH = 0.8

    train_transform = A.Compose([
    A.OneOf([
        A.RandomCrop(width=int(IMAGE_WIDTH * CROP_SCALE_WIDTH), height=int(IMAGE_HEIGHT * CROP_SCALE_HEIGHT), p=1),
        A.CenterCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, p=1)
    ], p=1),
    
    A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Sharpen(p=0.2),
    A.GaussNoise(var_limit=(10, 50),p=0.1),
    ],
        additional_targets={
            "radar_img": "mask",
            "ground_truth": "mask",
            "mmde_map": "mask"
        }
    )

    val_transform = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, p=1),
        ],
        additional_targets={
        "radar_img": "mask",
        "ground_truth": "mask",
        "mmde_map": "mask"
        }
    )

    import ast
    # read from file
    file_path = os.path.join("outputs", "mean_depth_dataset.txt")

    with open(file_path, mode="r") as f:
        lines = f.readlines()

    mean_line = lines[0].split()[1:]
    std_line = lines[1].split()[1:]

    MEAN = ast.literal_eval(" ".join(mean_line))
    STD = ast.literal_eval(" ".join(std_line))

    print("Mean:", MEAN)
    print("Std:", STD)

    image_folder = "../data/image"
    mmde_map_path = "../data/mmde_map"
    gt_path = "../data/gt_interp"
    radar_path ="../data/radar_png"

    image_paths = glob.glob(os.path.join(image_folder, '*.*'))
    mmde_map = glob.glob(os.path.join(mmde_map_path, '*.*'))
    ground_truth_paths = glob.glob(os.path.join(gt_path, '*.*'))
    radar_paths = glob.glob(os.path.join(radar_path, '*.*'))

    # Split the dataset into training and validation sets
    train_image_paths, val_image_paths, train_mmde_map, val_mmde_map, train_radar_paths, val_radar_paths, train_gt_paths, val_gt_paths = train_test_split(
        image_paths, mmde_map, radar_paths, ground_truth_paths, test_size=0.2, random_state=42)
    
    train_dataset = TrainDataLoader(image_paths=train_image_paths,
                                mmde_path = train_mmde_map,
                                radar_paths = train_radar_paths,
                                ground_truth_paths = train_gt_paths,
                                transform=train_transform,
                                mean=MEAN,
                                std=STD)

    val_dataset = ValDataLoader(image_paths=val_image_paths,
                                mmde_path = val_mmde_map,
                                radar_paths = val_radar_paths,
                                ground_truth_paths = val_gt_paths,
                                transform=val_transform,
                                )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    params = Params()

    model = fusion_net(params)

    # hyperparameterss
    lr = max_lr = 1e-3
    weight_decay = 1e-2
    num_epochs = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE, type(DEVICE)

    # INFO
    num_train_samples = len(train_dataloader.dataset)
    num_val_samples = len(val_dataloader.dataset)

    num_train_steps = len(train_dataloader)
    num_val_steps = len(val_dataloader)
    print(f"Number of samples: Train: {num_train_samples} | Val: {num_val_samples}")
    print(f"Number of steps: Train: {num_train_steps} | Val: {num_val_steps}")

    criterial = l1_loss()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr, # max_lr = 1e-3
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        div_factor=10, # initial lr = max_lr/div_factor
        pct_start=0.3, #decay start at 30% of the total epochs
        anneal_strategy='cos' # anneal strategy
    )

    # # callbacks
    early_stopping = EarlyStopping(patience=20, verbose=True)
    checkpoint = Checkpoint(
        root_dir="depth_experiment",
        experiment_name="deph_head",
        checkpoint_name="model_depth",
        save_model_only=True
    )

    torch.set_float32_matmul_precision('medium')

    fabric = Fabric(accelerator="gpu", precision="32") # precision = "32" / "16-mixed" / "bf16-mixed"

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    wandb.init(project="thesis", config={"learning_rate": scheduler.get_last_lr()[0], "epochs": num_epochs, "batch_size": train_dataloader.batch_size})

    # Main training loop with wandb integration
    start_epoch = 1

    for epoch in range(start_epoch, 100 + 1):
        # Training Phase
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterial,
            fabric=fabric,
            epoch=epoch,
            num_steps=len(train_dataloader),
            mode="train"
        )

        # Validation Phase
        val_loss = train_one_epoch(
            model=model,
            dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterial,
            fabric=fabric,
            epoch=epoch,
            num_steps=len(val_dataloader),
            mode="val",
            eval_step=5  # Adjust eval_step as needed
        )

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

            # Log epoch results to wandb
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

            # Early stopping and checkpoint logic
        # early_stopping(val_loss)
        # if early_stopping.save_current_checkpoint:
        #     checkpoint(epoch, train_loss, val_loss, model, optimizer, scheduler)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered. Ending training.")
        #     break

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()








