"""Module to evaluate the model performance"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from fish_net.fish import Position, get_fishes_positions
from fish_net.load_data import MIDLINE_POINTS, ZONE_SIZE
from fish_net.models import DEVICE


@dataclass
class Metrics:
    """Dataclass to store the metrics of the model"""

    head_average_distance: float | None = None
    head_precision: float | None = None
    head_recall: float | None = None
    head_f1: float | None = None
    head_loss: float | None = None
    midline_loss: float | None = None
    midline_distance: float | None = None
    midline_recall_1: float | None = None
    midline_recall_3: float | None = None
    midline_recall_5: float | None = None
    fish_max_distance: float | None = None
    fish_recall_1: float | None = None
    fish_recall_3: float | None = None
    fish_recall_5: float | None = None
    fish_malformed: float | None = None
    n_predictions: int | None = None
    n_annotations: int | None = None
    roll_loss: float | None = None
    roll_precision: float | None = None
    roll_recall: float | None = None
    roll_f1: float | None = None


def get_prediction_metrics(
    annotations: list[Position],
    predictions: list[Position],
) -> Metrics:
    """
    Get the metrics of the model
    :param annotations: list of manually annotated Positions
    :param predictions: list of predicted Positions
    :return: Metrics
    """
    distances = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    tol = 5

    fishes_positions_annot = annotations.copy()
    fishes_positions_pred = predictions.copy()

    for fish_position in fishes_positions_annot:
        if fishes_positions_pred:
            closest_pred_fish = min(fishes_positions_pred, key=fish_position.distance)
            distance_to_closest = fish_position.distance(closest_pred_fish)
            if distance_to_closest < tol:
                distances.append(distance_to_closest)
                true_positives += 1
                fishes_positions_pred.remove(closest_pred_fish)
                continue
        false_negatives += 1

    false_positives += len(fishes_positions_pred)
    if true_positives:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        metrics = Metrics(
            head_average_distance=float(np.mean(distances)),
            head_precision=precision,
            head_recall=recall,
            head_f1=2 * (precision * recall) / (precision + recall),
            n_predictions=len(predictions),
            n_annotations=len(annotations),
        )
    else:
        metrics = Metrics(
            head_average_distance=np.nan,
            head_precision=0,
            head_recall=0,
            head_f1=0,
            n_predictions=len(predictions),
            n_annotations=len(annotations),
        )
    return metrics


def display_predictions(images, labels, output):
    """Display 3 images, labels and predictions in a plt figure"""
    _, axs = plt.subplots(3, 3, figsize=(30, 30))
    for i in range(3):
        axs[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        axs[1, i].imshow(labels[i].squeeze().cpu().numpy(), vmin=0, vmax=1)
        axs[2, i].imshow(output[i].squeeze().cpu().detach().numpy(), vmin=0, vmax=1)
        axs[0, i].axis("off")
        axs[1, i].axis("off")
        axs[2, i].axis("off")
    plt.tight_layout()
    plt.show()


def evaluate_on_test(
    model, test_dataloader, head_loss_fn, midline_loss_fn, roll_loss_fn=None
) -> Metrics:
    """
    Evaluate the model on the test set
    :param model: torch.nn.Module
    :param test_dataloader: DataLoader
    :param loss_fn: torch.nn.Module
    """
    model.eval()
    image_index = 0
    head_metrics_list = []
    midline_metrics_list = []
    roll_metrics_list = []

    with torch.no_grad():
        for (
            images,
            head_labels,
            midline_labels,
            head_positions,
            _,
            rolling_states,
        ) in test_dataloader:
            images = images.to(DEVICE)
            head_labels = head_labels.to(DEVICE)
            midline_labels = midline_labels.to(DEVICE)
            head_positions = head_positions.to(DEVICE)
            rolling_states = rolling_states.to(DEVICE)
            head_output = model(images)
            head_loss = head_loss_fn(head_output, head_labels)

            head_output = F.sigmoid(head_output).squeeze(1)
            for image_pred in head_output:
                pred_fishes_positions = get_fishes_positions(image_pred.cpu().numpy())
                head_annotations = [
                    position.reversed
                    for position in test_dataloader.dataset.data[
                        image_index
                    ].annotations_head
                ]
                metrics = get_prediction_metrics(
                    head_annotations, pred_fishes_positions
                )
                metrics.head_loss = head_loss.item()
                head_metrics_list.append(metrics)
                image_index += 1

            if midline_loss_fn:
                midline_output = model.midline_forward(
                    head_positions, zone_size=ZONE_SIZE
                )
                midline_output = midline_output.contiguous().view(
                    midline_output.shape[0] * midline_output.shape[1],
                    midline_output.shape[2] * midline_output.shape[3],
                )

                midline_labels = midline_labels.view(
                    midline_labels.shape[0]
                    * midline_labels.shape[1]
                    * midline_labels.shape[2],
                    midline_labels.shape[3] * midline_labels.shape[4],
                )

                midline_labels = midline_labels[midline_labels.sum(dim=1) > 0]

                midline_loss = midline_loss_fn(midline_output, midline_labels)

                midline_labels_argmax = midline_labels.argmax(dim=1).cpu().numpy()
                midline_output_argmax = midline_output.argmax(dim=1).cpu().numpy()
                midline_labels_coords = np.column_stack(
                    np.unravel_index(midline_labels_argmax, (ZONE_SIZE, ZONE_SIZE))
                )
                midline_output_coords = np.column_stack(
                    np.unravel_index(midline_output_argmax, (ZONE_SIZE, ZONE_SIZE))
                )
                distances = np.sqrt(
                    ((midline_labels_coords - midline_output_coords) ** 2).sum(axis=1)
                )
                fish_distances = distances.reshape(-1, MIDLINE_POINTS)
                fish_coords = midline_output_coords.reshape(-1, MIDLINE_POINTS, 2)
                fish_coords_pairwise_distance = np.sqrt(
                    ((fish_coords[:, :-1] - fish_coords[:, 1:]) ** 2).sum(axis=2)
                )
                fish_coords_pairwise_distance_median = np.median(
                    fish_coords_pairwise_distance, axis=1
                )
                fish_malformed = np.any(
                    fish_coords_pairwise_distance
                    > (fish_coords_pairwise_distance_median[:, None] * 2),
                    axis=1,
                ).mean()

                midline_metrics_list.append(
                    Metrics(
                        midline_loss=midline_loss.item(),
                        midline_distance=distances.mean(),
                        midline_recall_1=(distances <= 1).mean(),
                        midline_recall_3=(distances <= 3).mean(),
                        midline_recall_5=(distances <= 5).mean(),
                        fish_max_distance=fish_distances.max(axis=1).mean(),
                        fish_recall_1=(fish_distances <= 1).all(axis=1).mean(),
                        fish_recall_3=(fish_distances <= 3).all(axis=1).mean(),
                        fish_recall_5=(fish_distances <= 5).all(axis=1).mean(),
                        fish_malformed=fish_malformed,
                    )
                )

            # ROLLING
            if roll_loss_fn is not None:
                # predict rolling states for valid fish only
                roll_output = model.classifier_forward()
                mask = head_positions.sum(dim=-1) != 0  # shape: (B, max_fish_per_frame)
                valid_roll_states = rolling_states[
                    mask
                ].float()  # shape: (num_valid_fish,)

                # shape of roll_output is (num_valid_fish, 1) => flatten to match
                roll_loss = roll_loss_fn(
                    roll_output.squeeze(-1), valid_roll_states.float()
                )

                predictions = (F.sigmoid(roll_output) > 0.5).float().squeeze(-1)
                # compute recall, precision and f1 for the roll prediction
                true_positives = (predictions * valid_roll_states).sum().item()
                false_positives = ((1 - valid_roll_states) * predictions).sum().item()
                false_negatives = (valid_roll_states * (1 - predictions)).sum().item()

                if true_positives:
                    roll_precision = true_positives / (true_positives + false_positives)
                    roll_recall = true_positives / (true_positives + false_negatives)
                    roll_f1 = (
                        2
                        * (roll_precision * roll_recall)
                        / (roll_precision + roll_recall)
                    )
                else:
                    roll_precision = np.nan
                    roll_recall = np.nan
                    roll_f1 = np.nan

                # store them in a 'Metrics' object
                roll_metrics_list.append(
                    Metrics(
                        roll_loss=roll_loss.item(),
                        roll_precision=roll_precision,
                        roll_recall=roll_recall,
                        roll_f1=roll_f1,
                    )
                )

    def average_metric(metric_list, metric_name):
        values = [getattr(metrics, metric_name) for metrics in metric_list]
        values = [value for value in values if not np.isnan(value)]
        return float(np.mean(values)) if values else np.nan

    model_metrics = Metrics(
        head_loss=average_metric(head_metrics_list, "head_loss"),
        head_precision=average_metric(head_metrics_list, "head_precision"),
        head_recall=average_metric(head_metrics_list, "head_recall"),
        head_f1=average_metric(head_metrics_list, "head_f1"),
        head_average_distance=average_metric(
            head_metrics_list, "head_average_distance"
        ),
        midline_loss=average_metric(midline_metrics_list, "midline_loss"),
        midline_distance=average_metric(midline_metrics_list, "midline_distance"),
        midline_recall_1=average_metric(midline_metrics_list, "midline_recall_1"),
        midline_recall_3=average_metric(midline_metrics_list, "midline_recall_3"),
        midline_recall_5=average_metric(midline_metrics_list, "midline_recall_5"),
        fish_max_distance=average_metric(midline_metrics_list, "fish_max_distance"),
        fish_recall_1=average_metric(midline_metrics_list, "fish_recall_1"),
        fish_recall_3=average_metric(midline_metrics_list, "fish_recall_3"),
        fish_recall_5=average_metric(midline_metrics_list, "fish_recall_5"),
        fish_malformed=average_metric(midline_metrics_list, "fish_malformed"),
        roll_loss=average_metric(roll_metrics_list, "roll_loss"),
        roll_precision=average_metric(roll_metrics_list, "roll_precision"),
        roll_recall=average_metric(roll_metrics_list, "roll_recall"),
        roll_f1=average_metric(roll_metrics_list, "roll_f1"),
    )

    return model_metrics
