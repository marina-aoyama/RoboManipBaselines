import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    DpStyleDatasetMixin,
    RmbData,
    convert_data_to_policy,
    get_skipped_data_seq,
)


class ManiFlowImageDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train maniflow policy with image."""

    def setup_variables(self):
        self.setup_dp_style_chunk()

    def setup_image_transforms(self):
        """
        Only the uint8-compatible spatial augmentations run here. Dtype conversion
        (uint8 -> float32) and the [-1, 1] rescale are deferred to the GPU (see
        TrainManiFlowPolicy), since doing them here would transfer 4x larger
        float32 image batches over PCIe on every step.
        """
        image_transform_list = []

        if self.model_meta_info["image"]["aug_erasing_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_erasing_scale"]
            image_transform_list.append(v2.RandomErasing(p=0.5 * scale))

        if self.model_meta_info["image"]["aug_color_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_color_scale"]
            image_transform_list.append(
                v2.ColorJitter(
                    brightness=0.4 * scale,
                    contrast=0.4 * scale,
                    saturation=0.4 * scale,
                    hue=0.05 * scale,
                )
            )

        if self.model_meta_info["image"]["aug_affine_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_affine_scale"]
            image_transform_list.append(
                v2.RandomAffine(
                    degrees=4.0 * scale,
                    translate=(0.05 * scale, 0.05 * scale),
                    scale=(1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
                )
            )

        if len(image_transform_list) == 0:
            image_transform_list.append(v2.Identity())
        self.image_transforms = v2.Compose(image_transform_list)

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(
            self.filenames[episode_idx],
            self.enable_rmb_cache,
            image_size=self.model_meta_info["data"]["image_size"],
        ) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip)[
                                time_idxes
                            ],
                            key,
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Convert to data structure of policy input and output
        data = {"obs": {}, "action": action_tensor}
        if len(self.model_meta_info["state"]["keys"]) > 0:
            data["obs"]["state"] = state_tensor
        for camera_idx, camera_name in enumerate(
            self.model_meta_info["image"]["camera_names"]
        ):
            data["obs"][DataKey.get_rgb_image_key(camera_name)] = images_tensor[
                camera_idx
            ]

        return data
