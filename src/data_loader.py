import os
import glob
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    MapTransform,
    Orientationd,
    Spacingd,
    ConcatItemsd,
)
from monai.data import PersistentDataset, DataLoader, Dataset, CacheDataset


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Converts the raw BraTS label map (values 1, 2, 4) into three
    overlapping binary channels representing the three clinically
    meaningful tumor sub-regions:
        - Tumor Core (TC):     label == 1 OR label == 4
        - Whole Tumor (WT):    label == 1 OR label == 4 OR label == 2
        - Enhancing Tumor (ET): label == 4
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # [1, 2, 4] -> [TC, WT, ET]
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d


def get_brats_dataloaders(data_dir="./data", cache_dir="./cache", batch_size=1):
    """
    Builds the TRAINING dataloader. Uses RandSpatialCropd to crop each
    volume down to a fixed 96x96x96 patch, which keeps GPU memory usage
    manageable during training (full volumes are ~240x240x155).
    """
    patient_folders = [f for f in sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*"))) if os.path.isdir(f)]

    data_dicts = []
    for folder in patient_folders:
        t1 = glob.glob(os.path.join(folder, "*_t1.nii.gz"))
        t1ce = glob.glob(os.path.join(folder, "*_t1ce.nii.gz"))
        t2 = glob.glob(os.path.join(folder, "*_t2.nii.gz"))
        flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
        seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))

        if all([t1, t1ce, t2, flair, seg]):
            data_dicts.append({
                "t1": t1[0], "t1ce": t1ce[0], "t2": t2[0], "flair": flair[0],
                "label": seg[0],
            })

    if not data_dicts:
        raise FileNotFoundError(
            f"No complete patient folders found under {data_dir}. Expected "
            f"subfolders named BraTS2021_* each containing t1, t1ce, t2, "
            f"flair, and seg .nii.gz files. See README.md for where to get "
            f"the BraTS dataset."
        )

    img_keys = ["t1", "t1ce", "t2", "flair"]
    all_keys = img_keys + ["label"]

    train_transform = Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),

        # Crop BEFORE concatenation to keep metadata intact
        RandSpatialCropd(keys=all_keys, roi_size=[96, 96, 96], random_size=False),

        # Concatenate the 4 modalities into one 4-channel "image" and
        # convert the raw label into the 3 target regions
        ConcatItemsd(keys=img_keys, name="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    train_ds = Dataset(data=data_dicts, transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader


def get_full_brain_dataloader(data_dir="./data"):
    """
    Loads the FULL, uncropped brain volume for evaluation/inference
    (used with Sliding Window Inference in test_brats.py and
    evaluate_metrics.py) -- no RandSpatialCropd here, since we want a
    prediction over the entire real brain, not a random training crop.
    """
    patient_folders = [f for f in sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*"))) if os.path.isdir(f)]

    data_dicts = []
    for folder in patient_folders:
        t1 = glob.glob(os.path.join(folder, "*_t1.nii.gz"))
        t1ce = glob.glob(os.path.join(folder, "*_t1ce.nii.gz"))
        t2 = glob.glob(os.path.join(folder, "*_t2.nii.gz"))
        flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
        seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))

        if all([t1, t1ce, t2, flair, seg]):
            data_dicts.append({
                "t1": t1[0], "t1ce": t1ce[0], "t2": t2[0], "flair": flair[0],
                "label": seg[0],
            })

    if not data_dicts:
        raise FileNotFoundError(
            f"No complete patient folders found under {data_dir}. See README.md."
        )

    img_keys = ["t1", "t1ce", "t2", "flair"]
    all_keys = img_keys + ["label"]

    test_transform = Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        ConcatItemsd(keys=img_keys, name="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    test_ds = Dataset(data=data_dicts, transform=test_transform)
    # batch_size MUST be 1 -- full brain volumes vary and are large
    return DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    loader = get_brats_dataloaders()
    print("Testing training data loader...")
    for batch_data in loader:
        print(f"Image shape: {batch_data['image'].shape}")  # [1, 4, 96, 96, 96]
        print(f"Label shape: {batch_data['label'].shape}")   # [1, 3, 96, 96, 96]
        break
