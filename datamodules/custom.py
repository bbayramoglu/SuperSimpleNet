from pathlib import Path

import albumentations as A
import pandas as pd
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod
from pandas import DataFrame

from datamodules.base.datamodule import SSNDataModule, BgMask
from datamodules.base.dataset import SSNDataset


class CustomDataset(SSNDataset):
    """Generic dataset that follows the MVTec folder structure."""

    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        split: Split,
        supervised: bool,
        flips: bool,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            supervised=supervised,
            debug=debug,
        )

    def _gather_samples(self, img_dir: Path, split: Split) -> list[list[str | int]]:
        samples: list[list[str | int]] = []
        for img_path in sorted(img_dir.glob("**/*")):
            if not img_path.is_file():
                continue
            stem = img_path.stem
            mask_path = img_dir.parent.parent / "ground_truth" / img_dir.name / f"{stem}.png"
            if not mask_path.exists():
                mask_path = ""
            label = LabelName.NORMAL if img_dir.name == "good" else LabelName.ABNORMAL
            samples.append(
                [
                    str(self.root),
                    stem,
                    split.value,
                    str(img_path),
                    str(mask_path),
                    label,
                ]
            )
        return samples

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        if self.split == Split.TRAIN:
            base_dir = self.root / "train"
        else:
            base_dir = self.root / "test"

        all_samples = []
        for sub in sorted(base_dir.iterdir()):
            if not sub.is_dir():
                continue
            all_samples.extend(self._gather_samples(sub, self.split))

        samples = DataFrame(
            all_samples,
            columns=["path", "sample_id", "split", "image_path", "mask_path", "label_index"],
        )
        samples["label_index"] = samples["label_index"].astype(int)

        normal_samples = samples.loc[samples.label_index == LabelName.NORMAL].reset_index()
        anomalous_samples = samples.loc[samples.label_index == LabelName.ABNORMAL].reset_index()

        return normal_samples, anomalous_samples


class Custom(SSNDataModule):
    """Datamodule for a custom dataset."""

    def __init__(
        self,
        root: Path | str,
        supervised: bool,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        print(f"Resolution set to: {image_size}")
        super().__init__(
            root=root,
            supervised=supervised,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
            mask_bg=BgMask.NONE,
        )

        self.train_data = CustomDataset(
            transform=self.transform_train,
            split=Split.TRAIN,
            root=Path(root),
            supervised=supervised,
            flips=flips,
            normal_flips=normal_flips,
            debug=debug,
        )
        self.test_data = CustomDataset(
            transform=self.transform_eval,
            split=Split.TEST,
            root=Path(root),
            supervised=supervised,
            flips=flips,
            normal_flips=False,
            debug=debug,
        )

