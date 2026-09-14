"""
This script trains a model from scratch using the supervised dataset.
"""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fish_net.load_data import (
    HEAD_ANNOTATIONS_FOLDER,
    HEAD_SIGMA_END,
    HEAD_SIGMA_START,
    MAX_FISH_PER_FRAME,
    MIDLINE_ANNOTATIONS_FOLDER,
    MIDLINE_POINTS,
    MIDLINE_SIGMA_END,
    MIDLINE_SIGMA_START,
    ZONE_SIZE,
    SupervisedDataset,
)
from fish_net.metrics import evaluate_on_test
from fish_net.models import DEVICE, FishNet

MODEL_NAME = "fishnet"
SAVE_DIR = "outputs"
SEED = 42
EPOCHS = 75
BONUS_EPOCH = 25
PATIENCE = 50
BATCH_SIZE = 2
KERNEL_SIZE = 3
BACKBONE_CHANNELS = 64
MIDLINE_CHANNELS = 64
CLASSIFIER_CHANNELS = 64
BACKBONE_LAYERS = 6
MIDLINE_LAYERS = 6
MIDLINE_ALPHA = 0.01
ROLL_ALPHA = 1
LEARNING_RATE = 0.001
HEAD_LOSS_WEIGHT = 0.001
HEAD_GAMMA = (HEAD_SIGMA_END / HEAD_SIGMA_START) ** (1 / EPOCHS)
MIDLINE_GAMMA = (MIDLINE_SIGMA_END / MIDLINE_SIGMA_START) ** (1 / EPOCHS)
AUGMENT = True

# ANNOTATIONS_FOLDER = HEAD_ANNOTATIONS_FOLDER
ANNOTATIONS_FOLDER = MIDLINE_ANNOTATIONS_FOLDER
if ANNOTATIONS_FOLDER == HEAD_ANNOTATIONS_FOLDER:
    TEST_FOLDERS = ["test-experiments"]
elif ANNOTATIONS_FOLDER == MIDLINE_ANNOTATIONS_FOLDER:
    TEST_FOLDERS = [
        "2023_03_07_CPF_CPO_2PAM_n=1_1h",
        "2023_06_21_ToCP_CBDP_4h_n=4",
        "18_03_27_ToCP",
        "22_04_26_WT_5_dpf_CPO_2PAM_n=1",
        "2021_06_29_TBT_TCS_T007_n=2",
    ]
    print(f"Test folders: {TEST_FOLDERS}")


@dataclass
class EarlyStopping:
    """
    Dataclass to store the early stopping parameters
    """

    epochs_without_improvement: int = 0
    best_weights: dict | None = None
    best_loss: float = np.inf
    custom_score: float = -np.inf


def get_dataloaders(
    annotations_folder: str, test_experiments: list[str]
) -> tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the training and testing sets
    """
    train_experiments = [
        experiment
        for experiment in os.listdir(annotations_folder)
        if experiment not in test_experiments
    ]

    train_dataset = SupervisedDataset(
        annotations_folder=annotations_folder,
        experiments=train_experiments,
        head_sigma=HEAD_SIGMA_START,
        midline_sigma=MIDLINE_SIGMA_START,
        augment=AUGMENT,
    )
    test_dataset = SupervisedDataset(
        annotations_folder=annotations_folder,
        experiments=test_experiments,
        head_sigma=HEAD_SIGMA_START,
        midline_sigma=MIDLINE_SIGMA_START,
        augment=False,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} samples in training dataset")
    print(f"Loaded {len(test_dataset)} samples in testing dataset")

    return train_dataloader, test_dataloader


def set_seed(seed):
    """Set the seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class BCEWithLogitsWeightedLoss(nn.Module):
    """Weighted binary cross-entropy loss"""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): The raw model outputs (before applying sigmoid) of shape (N, *).
            labels (torch.Tensor): The ground truth labels of shape (N, *), with values in [0, 1].

        Returns:
            torch.Tensor: The computed weighted binary cross-entropy loss.
        """
        # Ensure the labels are in float format
        labels = labels.float()

        # Compute foreground and background sizes
        labels_true = labels > 0.5
        total_elements = labels.numel()
        foreground_size = labels_true.sum().item()
        background_size = total_elements - foreground_size

        # Handle edge case: no foreground or no background
        if foreground_size == 0 or background_size == 0:
            raise ValueError(
                "Labels must have both foreground and background elements."
            )

        # Compute weights
        weights = torch.ones_like(labels, device=labels.device)
        weights[labels_true] = (background_size / foreground_size) * self.weight
        weights /= weights.sum() / total_elements

        # Compute weighted binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)

        return loss


def train(annotation_folder):
    """
    Training loop for the supervised learning.
    :return: None
    """
    set_seed(SEED)

    # Load the dataset
    train_dataloader, test_dataloader = get_dataloaders(
        annotations_folder=annotation_folder,
        test_experiments=TEST_FOLDERS,
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Build the model from scratch
    model = FishNet(
        midline_points=MIDLINE_POINTS,
        backbone_channels=BACKBONE_CHANNELS,
        midline_channels=MIDLINE_CHANNELS,
        classifier_channels=CLASSIFIER_CHANNELS,
        backbone_layers=BACKBONE_LAYERS,
        midline_layers=MIDLINE_LAYERS,
        kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    head_loss_fn = BCEWithLogitsWeightedLoss(weight=HEAD_LOSS_WEIGHT)
    midline_loss_fn = nn.CrossEntropyLoss()
    roll_loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS + BONUS_EPOCH, eta_min=LEARNING_RATE / 100
    )
    paramaters_cnt = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {paramaters_cnt:,}")
    early_stopping = EarlyStopping()

    fishes_in_train = sum(
        min(MAX_FISH_PER_FRAME, len(data.annotations_head))
        for data in train_dataloader.dataset.data
    )
    fishes_in_test = sum(
        min(MAX_FISH_PER_FRAME, len(data.annotations_head))
        for data in test_dataloader.dataset.data
    )
    wandb.init(
        mode=os.environ.get("WANDB_MODE", "disabled"),
        project="fishnet",
        config={
            "epochs": EPOCHS + BONUS_EPOCH,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "max_fish_per_frame": MAX_FISH_PER_FRAME,
            "fishes_in_train": fishes_in_train,
            "fishes_in_test": fishes_in_test,
            "learning_rate": LEARNING_RATE,
            "kernel_size": KERNEL_SIZE,
            "backbone_channels": BACKBONE_CHANNELS,
            "midline_channels": MIDLINE_CHANNELS,
            "backbone_layers": BACKBONE_LAYERS,
            "midline_layers": MIDLINE_LAYERS,
            "head_sigma_start": HEAD_SIGMA_START,
            "head_sigma_end": HEAD_SIGMA_END,
            "midline_sigma_start": MIDLINE_SIGMA_START,
            "midline_sigma_end": MIDLINE_SIGMA_END,
            "midline_alpha": MIDLINE_ALPHA,
            "augment": AUGMENT,
            "zone_size": ZONE_SIZE,
            "model": type(model).__name__,
            "paramaters_cnt": paramaters_cnt,
            "train_samples": len(train_dataloader.dataset),
            "test_samples": len(test_dataloader.dataset),
        },
    )

    for epoch in range(EPOCHS + BONUS_EPOCH):
        running_head_loss = 0
        running_midline_loss = 0
        running_roll_loss = 0
        samples_cnt = 0
        with tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{EPOCHS + BONUS_EPOCH}",
            ascii=" >=",
            leave=False,
        ) as pbar:
            pbar.set_postfix({"loss": None, "midline_loss": None, "roll_loss": None})

            model.train()
            for batch_idx, (
                images,
                head_labels,
                midline_labels,
                head_positions,
                _,
                rolling_states,
            ) in enumerate(pbar):
                images = images.to(DEVICE)
                head_labels = head_labels.to(DEVICE)
                midline_labels = midline_labels.to(DEVICE)
                head_positions = head_positions.to(DEVICE)
                rolling_states = rolling_states.to(DEVICE)
                optimizer.zero_grad()
                head_output = model(images)
                head_loss = head_loss_fn(head_output, head_labels)

                if ANNOTATIONS_FOLDER == MIDLINE_ANNOTATIONS_FOLDER:
                    midline_output = model.midline_forward(
                        head_positions, zone_size=ZONE_SIZE
                    )

                    midline_labels = midline_labels.view(
                        midline_labels.shape[0] * midline_labels.shape[1],
                        midline_labels.shape[2],
                        midline_labels.shape[3],
                        midline_labels.shape[4],
                    )

                    midline_labels = midline_labels[
                        midline_labels.sum(dim=(1, 2, 3)) > 0
                    ]

                    midline_output_view = midline_output.view(
                        midline_output.shape[0] * midline_output.shape[1],
                        midline_output.shape[2] * midline_output.shape[3],
                    )

                    midline_labels_view = midline_labels.view(
                        midline_labels.shape[0] * midline_labels.shape[1],
                        midline_labels.shape[2] * midline_labels.shape[3],
                    )

                    midline_loss = midline_loss_fn(
                        midline_output_view, midline_labels_view
                    )
                    running_midline_loss += midline_loss.item() * images.size(0)

                # ROLLING
                roll_output = model.classifier_forward()
                mask = head_positions.sum(dim=-1) != 0
                valid_roll_states = rolling_states[mask].float()
                roll_loss = roll_loss_fn(roll_output.squeeze(-1), valid_roll_states)
                running_roll_loss += roll_loss.item() * images.size(0)

                head_loss.backward(retain_graph=True)
                (midline_loss * MIDLINE_ALPHA).backward()
                (roll_loss * ROLL_ALPHA).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_head_loss += head_loss.item() * images.size(0)
                samples_cnt += images.size(0)

                if (batch_idx + 1) % 20 == 0:
                    with torch.no_grad():
                        head_output_arr = (
                            F.sigmoid(head_output).cpu().detach().numpy() > 0.5
                        )
                        head_labels_arr = head_labels.cpu().detach().numpy() > 0.5
                        true_positives = (head_output_arr & head_labels_arr).sum()
                        false_positives = (head_output_arr & ~head_labels_arr).sum()
                        false_negatives = (~head_output_arr & head_labels_arr).sum()
                        if true_positives == 0:
                            head_f1_px = 0
                            head_precision_px = 0
                            head_recall_px = 0
                        else:
                            head_f1_px = (
                                2
                                * true_positives
                                / (
                                    2 * true_positives
                                    + false_positives
                                    + false_negatives
                                )
                            )
                            head_precision_px = true_positives / (
                                true_positives + false_positives
                            )
                            head_recall_px = true_positives / (
                                true_positives + false_negatives
                            )

                        midline_labels_argmax = (
                            midline_labels_view.argmax(dim=1).cpu().numpy()
                        )
                        midline_output_argmax = (
                            midline_output_view.argmax(dim=1).cpu().numpy()
                        )
                        midline_labels_coords = np.column_stack(
                            np.unravel_index(
                                midline_labels_argmax, (ZONE_SIZE, ZONE_SIZE)
                            )
                        )
                        midline_output_coords = np.column_stack(
                            np.unravel_index(
                                midline_output_argmax, (ZONE_SIZE, ZONE_SIZE)
                            )
                        )
                        distances = np.sqrt(
                            ((midline_labels_coords - midline_output_coords) ** 2).sum(
                                axis=1
                            )
                        )
                        predictions = (F.sigmoid(roll_output) > 0.5).float().squeeze(-1)
                        # compute recall, precision and f1 for the roll prediction
                        true_positives = (predictions * valid_roll_states).sum().item()
                        false_positives = (
                            ((1 - valid_roll_states) * predictions).sum().item()
                        )
                        false_negatives = (
                            (valid_roll_states * (1 - predictions)).sum().item()
                        )
                        if true_positives:
                            roll_precision = true_positives / (
                                true_positives + false_positives
                            )
                            roll_recall = true_positives / (
                                true_positives + false_negatives
                            )
                            roll_f1 = (
                                2
                                * (roll_precision * roll_recall)
                                / (roll_precision + roll_recall)
                            )
                        else:
                            roll_precision = 0
                            roll_recall = 0
                            roll_f1 = 0
                        wandb.log(
                            {
                                "train_head_loss": head_loss.item(),
                                "train_head_precision_px": head_precision_px,
                                "train_head_recall_px": head_recall_px,
                                "train_head_f1_px": head_f1_px,
                                "train_midline_loss": midline_loss.item(),
                                "train_midline_distance": distances.mean(),
                                "train_midline_recall@1": (distances <= 1).mean(),
                                "train_midline_recall@3": (distances <= 3).mean(),
                                "train_midline_recall@5": (distances <= 5).mean(),
                                "train_roll_loss": roll_loss.item(),
                                "train_roll_precision": roll_precision,
                                "train_roll_recall": roll_recall,
                                "train_roll_f1": roll_f1,
                                "epoch": epoch + 1,
                            },
                            step=epoch * len(train_dataloader.dataset) + samples_cnt,
                        )

                pbar.set_postfix(
                    {
                        "loss": f"{running_head_loss / samples_cnt:.5f}",
                        "midline_loss": f"{running_midline_loss / samples_cnt:.5f}",
                        "roll_loss": f"{running_roll_loss / samples_cnt:.5f}",
                    }
                )

            # validation
            with torch.no_grad():
                metrics = evaluate_on_test(
                    model, test_dataloader, head_loss_fn, midline_loss_fn, roll_loss_fn
                )
            print(
                f"\nEpoch {epoch + 1}/{EPOCHS + BONUS_EPOCH} - "
                f"Training loss: (head: {running_head_loss / samples_cnt:.5f}, "
                f"midline: {running_midline_loss / samples_cnt:.5f}, "
                f"rolling: {running_roll_loss / samples_cnt:.5f}), "
                f"Validation loss: (head: {metrics.head_loss:.5f}, "
                f"midline: {metrics.midline_loss:.5f}, "
                f"rolling: {metrics.roll_loss:.5f}), "
                f"Head f1: {metrics.head_f1:.4f}, "
                f"Fish recall@3: {metrics.fish_recall_3:.4f}, "
                f"Fish recall@5: {metrics.fish_recall_5:.4f}, "
                f"Roll f1: {metrics.roll_f1:.4f}"
            )
            wandb.log(
                {
                    "val_head_loss": metrics.head_loss,
                    "val_head_precision": metrics.head_precision,
                    "val_head_recall": metrics.head_recall,
                    "val_head_f1": metrics.head_f1,
                    "val_midline_loss": metrics.midline_loss,
                    "val_roll_loss": metrics.roll_loss,
                    "val_roll_precision": metrics.roll_precision,
                    "val_roll_recall": metrics.roll_recall,
                    "val_roll_f1": metrics.roll_f1,
                    "val_midline_distance": metrics.midline_distance,
                    "val_midline_recall@1": metrics.midline_recall_1,
                    "val_midline_recall@3": metrics.midline_recall_3,
                    "val_midline_recall@5": metrics.midline_recall_5,
                    "val_fish_max_distance": metrics.fish_max_distance,
                    "val_fish_recall@1": metrics.fish_recall_1,
                    "val_fish_recall@3": metrics.fish_recall_3,
                    "val_fish_recall@5": metrics.fish_recall_5,
                    "val_fish_malformed": metrics.fish_malformed,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "midline_sigma": train_dataloader.dataset.midline_sigma,
                    "epoch": epoch + 1,
                },
                step=epoch * len(train_dataloader.dataset) + samples_cnt,
            )
            scheduler.step()
            # Custom early stopping score: maximize recall, minimize malformed
            assert metrics.fish_recall_3 is not None
            assert metrics.fish_recall_5 is not None
            assert metrics.fish_malformed is not None
            custom_score = (
                metrics.fish_recall_3 + metrics.fish_recall_5
            ) / 2 - metrics.fish_malformed

            if custom_score > early_stopping.custom_score:
                early_stopping.custom_score = custom_score
                early_stopping.epochs_without_improvement = 0
                early_stopping.best_weights = model.state_dict()
                torch.save(
                    early_stopping.best_weights,
                    os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt"),
                )
                print(f"Saving model at epoch {epoch}")
            else:
                early_stopping.epochs_without_improvement += 1

            if epoch < EPOCHS:
                train_dataloader.dataset.update_midline_annotations(gamma=MIDLINE_GAMMA)
                train_dataloader.dataset.update_head_annotations(gamma=HEAD_GAMMA)
                test_dataloader.dataset.update_midline_annotations(gamma=MIDLINE_GAMMA)
                test_dataloader.dataset.update_head_annotations(gamma=HEAD_GAMMA)

            if early_stopping.epochs_without_improvement == PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Save the last model as last_epoch
    last_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_last_epoch.pt")
    torch.save(model.state_dict(), last_model_path)

    # Log both best and last models as wandb artifacts
    artifact = wandb.Artifact("fishnet-models", type="model")
    artifact.add_file(os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt"))  # best
    artifact.add_file(last_model_path)  # last
    wandb.log_artifact(artifact)

    wandb.finish()


def main():
    global SAVE_DIR

    parser = argparse.ArgumentParser(
        description="Train FishNet on an annotation folder"
    )
    parser.add_argument(
        "--annotations-folder",
        default=MIDLINE_ANNOTATIONS_FOLDER,
        help="Folder containing experiment/video/annotations.json records",
    )
    parser.add_argument("--output-dir", default=SAVE_DIR)
    args = parser.parse_args()
    SAVE_DIR = args.output_dir
    train(args.annotations_folder)


if __name__ == "__main__":
    main()
