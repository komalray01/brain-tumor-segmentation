import os
import torch
import numpy as np
import nibabel as nib
from monai.inferers import SlidingWindowInferer

from data_loader import get_full_brain_dataloader
from train_brats import BraTSModel


def run_evaluation(checkpoint_path="./checkpoints/brats-best.ckpt"):
    """
    Runs inference on a full, uncropped brain volume using Sliding Window
    Inference: the model only ever saw 96x96x96 crops during training, so
    this slides a 96x96x96 window across the entire real brain (with 50%
    overlap between neighboring windows, blended smoothly), producing a
    prediction for the full 240x240x155 volume without needing that much
    GPU memory at once.

    Exports the prediction, the ground truth, and the FLAIR volume as
    NIfTI files for later 3D visualization (see render_interactive_3d.py).
    """
    print("Loading trained model...")
    model = BraTSModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Fetching raw, uncropped full brain scan...")
    data_loader = get_full_brain_dataloader()

    batch = next(iter(data_loader))
    images = batch["image"].to(model.device)
    labels = batch["label"]

    print(f"Input shape into model: {images.shape}")  # ~ [1, 4, 240, 240, 155]

    print("Running Sliding Window Inference across the entire brain...")
    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96),
        sw_batch_size=4,
        overlap=0.5,
    )

    with torch.no_grad():
        outputs = inferer(images, model)
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.2).float()

    print(f"Full brain processed. Final prediction shape: {preds.shape}")

    pred_data = preds[0].cpu().numpy()
    prediction_count = np.sum(pred_data)
    print(f"Total voxel detections in prediction: {int(prediction_count)}")

    pred_file = "omni_med_prediction.nii.gz"
    gt_file = "omni_med_ground_truth.nii.gz"
    flair_file = "omni_med_flair.nii.gz"

    for file_path in [pred_file, gt_file, flair_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

    if prediction_count == 0:
        print("WARNING: 0 tumor voxels predicted. Check the model checkpoint "
              "and the > 0.2 sigmoid threshold in this function if this is "
              "unexpected.")
        return

    # Save prediction: collapse the 3 channels (TC, WT, ET) into one
    # multi-valued volume for easier visualization
    final_3d_volume = np.zeros(pred_data.shape[1:], dtype=np.uint8)
    final_3d_volume[pred_data[0] == 1] = 1  # Whole Tumor
    final_3d_volume[pred_data[1] == 1] = 2  # Tumor Core
    final_3d_volume[pred_data[2] == 1] = 3  # Enhancing Tumor
    nib.save(nib.Nifti1Image(final_3d_volume, affine=np.eye(4)), pred_file)

    # Save ground truth in the same encoding for direct visual comparison
    gt_data = labels[0].cpu().numpy()
    gt_volume = np.zeros(gt_data.shape[1:], dtype=np.uint8)
    gt_volume[gt_data[0] == 1] = 1
    gt_volume[gt_data[1] == 1] = 2
    gt_volume[gt_data[2] == 1] = 3
    nib.save(nib.Nifti1Image(gt_volume, affine=np.eye(4)), gt_file)

    # Save the FLAIR channel as brain context for the 3D visualization
    flair_data = images[0, 3].cpu().numpy()
    flair_normalized = (
        (flair_data - flair_data.min()) / (flair_data.max() - flair_data.min() + 1e-8) * 255
    ).astype(np.uint8)
    nib.save(nib.Nifti1Image(flair_normalized, affine=np.eye(4)), flair_file)

    print("Done -- all NIfTI files exported successfully:")
    print(f"  {pred_file}, {gt_file}, {flair_file}")


if __name__ == "__main__":
    run_evaluation()
