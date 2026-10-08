import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer

from data_loader import get_full_brain_dataloader
from train_brats import BraTSModel


def calculate_metrics(checkpoint_path="./checkpoints/brats-best.ckpt", num_patients=5):
    """
    Computes the two headline evaluation metrics on held-out full-brain
    volumes:
      - Mean Dice Score: volume overlap between prediction and ground
        truth (2 x overlap / total volume of both). 1.0 = perfect overlap.
      - HD95 (95th-percentile Hausdorff Distance): boundary precision in
        real millimeters, ignoring the single most extreme outlier point.
        Complements Dice, which only measures overlap volume, not edge
        accuracy -- important since surgical planning cares about precise
        boundaries, not just approximate volume.
    """
    print("Loading model for quantitative evaluation...")
    model = BraTSModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_full_brain_dataloader()

    # include_background=False -- only score the 3 tumor classes, not the
    # (huge, trivially easy) empty background
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)

    print(f"Running evaluation on up to {num_patients} patients...")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_patients:
                break

            images = batch["image"].to(model.device)
            labels = batch["label"].to(model.device)

            while len(labels.shape) > 5:
                labels = labels.squeeze(2)

            outputs = inferer(images, model)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)

            print(f"Scored patient {i + 1}")

    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hd95_metric.aggregate().item()

    print("\n" + "=" * 40)
    print("QUANTITATIVE RESULTS")
    print("=" * 40)
    print(f"Mean Dice Score (DSC):     {mean_dice:.4f} (closer to 1.0 is better)")
    print(f"Hausdorff Distance (HD95): {mean_hd95:.2f} mm (lower is better)")
    print("=" * 40)

    return mean_dice, mean_hd95


if __name__ == "__main__":
    calculate_metrics()
