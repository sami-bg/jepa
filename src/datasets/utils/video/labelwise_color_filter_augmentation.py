from __future__ import annotations
import logging
import torch
from toolz import sliding_window
from functools import cache
import multiprocessing as mp
import pandas as pd
import csv
import colorsys
import os 
import json 

logger = logging.getLogger()
COLOR = tuple[float, float, float]
HUE   = int

@cache
def warn_once(x: str): logger.warning(x)


def max_gap_elts_radial(angles: list[HUE]) -> tuple[HUE, HUE]:
    # NOTE return the two angles (in [0,360)) which define the largest circular gap.
    # The second angle may exceed 360 if the wrap-around is used, so that
    # b - a = largest gap in a consistent ascending direction.
    
    assert len(angles) >= 2
    
    angles = sorted(angles)
    
    max_gap = -1
    best_pair = (0, 0)
    
    for i1, i2 in sliding_window(2, angles):
        gap = i2 - i1 # always non-negative
        if gap > max_gap:
            max_gap = gap
            best_pair = (i1, i2)
    
    wrap_gap = (angles[0] + 360) - angles[-1]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        best_pair = (angles[-1], angles[0] + 360)
    
    return best_pair



class LabelwiseColorFilterAugmentation:
    """
    An instance-level augmentation class that manages label→color mappings
    by reading/writing a JSON file at init time. Each instance corresponds
    to one split (e.g., 'train', 'val', etc.).
    
    NOTE: If multiple workers each create an instance, they'll each run __init__,
    potentially reading/writing the same file. That might be okay if you don't
    add new labels after the first pass. But if you do, watch out for concurrency.
    """

    # Where to store JSON for each split
    SPLIT_TO_JSON_PATH = {
        'train':      'label_colors_train.json',
        'val':        'label_colors_val.json',
        'eval':       'label_colors_eval.json',
        'test':       'label_colors_test.json',
        'distracted': 'label_colors_distracted.json',
    }

    # Starting hue for each split (just an example)
    SPLIT_TO_HUE_START = {
        'train': 0,
        'val': 0,
        'eval': 0,
        'test': 0,
        'distracted': 180,
    }

    def __init__(self, 
                 split="train", 
                 alpha: float = 0.3, 
                 normalize_fn=None, 
                 labels: list = None,
                 debug_file_prefix: str = ''):
        """
        :param split: which split, e.g. "train", "val", etc.
        :param alpha: some parameter for your transform
        :param normalize_fn: optional normalization function
        :param labels: a list of labels you want to ensure are assigned colors.
                       If None, we'll just load from file without adding new ones.
        """
        print(f'Initializing LabelwiseColorFilterAugmentation with {split=} {labels=}')
        assert split in self.SPLIT_TO_JSON_PATH, f"Invalid split={split}."
        self.debug_file_prefix = debug_file_prefix
        self.split = split
        self.alpha = alpha
        assert 0 <= self.alpha <= 1.0
        self.normalize_fn = normalize_fn

        # This object-level dictionary: label -> (R, G, B) in [0,1]
        self.label_to_color = {}
        # For convenience, also store label -> hue (in degrees) if you like
        self.label_to_hue = {}

        # 1. Load from JSON if it exists
        self.json_path = self.debug_file_prefix + self.SPLIT_TO_JSON_PATH[split]
        if os.path.exists(self.json_path):
            self._load_from_json()
        else:
            print(f"No existing color JSON found for split='{split}', creating a new one.")

        # 2. If user provided a list of labels, ensure each has a color
        if labels is not None:
            self.init_color_assignments_for_labels(labels)
            # Optionally, write the file back if new colors were added
            self._save_to_json()
        print(f'finished init ')

    def _load_from_json(self):
        """Loads existing {label -> [R,G,B]} mapping from disk, populates self.label_to_color."""
        with open(self.json_path, 'r') as f:
            saved_data = json.load(f)  # label -> [r_float, g_float, b_float]
        for lbl, rgb_list in saved_data.items():
            self.label_to_color[lbl] = tuple(rgb_list)
            # Recompute hue if needed
            (r, g, b) = rgb_list
            hsv = colorsys.rgb_to_hsv(r, g, b)  # (h in [0,1], s, v)
            hue_deg = hsv[0] * 360.0
            self.label_to_hue[lbl] = hue_deg

    def _save_to_json(self):
        """Writes {label -> [R,G,B]} to disk."""
        with open(self.json_path, 'w') as f:
            # Convert (r, g, b) tuples to lists
            data_out = {lbl: list(rgb) for lbl, rgb in self.label_to_color.items()}
            json.dump(data_out, f)

    def init_color_assignments_for_labels(self, labels: list):
        """
        Ensures each label in 'labels' has a color assigned. If any are missing,
        we generate a new hue placement, store in self.label_to_color/hue,
        then call _save_to_json() at the end.
        """
        changed = False  # track if we add new colors

        start_hue = self.SPLIT_TO_HUE_START[self.split]
        if self.split == "distracted":
            # breakpoint()
            pass
        for label in labels:
            if label in self.label_to_color:
                continue  # already assigned

            changed = True

            num_colors = len(self.label_to_color)
            if num_colors == 0:
                hue = start_hue
            elif num_colors == 1:
                # place second color on opposite side of color wheel
                hue = (start_hue + 180) % 360
            else:
                # find biggest gap
                existing_hues = list(self.label_to_hue.values())
                hue1, hue2 = max_gap_elts_radial(existing_hues)
                hue = (hue1 + hue2) / 2
                hue %= 360

            hsv_h = hue / 360.0
            (r, g, b) = colorsys.hsv_to_rgb(hsv_h, 1.0, 1.0)
            self.label_to_color[label] = (r, g, b)
            self.label_to_hue[label] = hue

        if changed:
            self._save_to_json()

    def assign_to_color(self, label: str):
        """
        Returns the (R,G,B) in [0,1] for a given label. If the label wasn't
        yet assigned, you can decide whether to auto-assign or raise an error.
        """
        if label not in self.label_to_color:
            # Auto-assign if you want:
            self.init_color_assignments_for_labels([label])
        return self.label_to_color[label]


    def _augment_frame_with_color(self, frame_CHW: torch.Tensor, color: torch.Tensor) -> torch.Tensor:

        tinted_frame = (1 - self.alpha) * frame_CHW + self.alpha * color        
        return torch.clamp(tinted_frame, 0, 1)
    

    def augment_video(self, video_CTHW: torch.Tensor, label: torch.Tensor | int | float) -> torch.Tensor:
        if label is None:
            warn_once(f'Received no label for LabelwiseColorFilterAugmentation')
            return video_CTHW
        
        if isinstance(label, torch.Tensor):
            label = label.item()

        T = video_CTHW.shape[1]
        if self.normalize_fn:
            video_CTHW = self.normalize_fn(video_CTHW)

        color = self.assign_to_color(label)
        color_vid_frames = torch.tensor(color, device=video_CTHW.device, dtype=video_CTHW.dtype)\
            .view(3, 1, 1)\
            .unsqueeze(0)\
            .repeat(T,1,1,1)

        label = str(label)
        for frame_idx in range(T):
            frame = video_CTHW[:, frame_idx, ::]
            color = color_vid_frames[frame_idx, ::]
            video_CTHW[:, frame_idx, ::] = self._augment_frame_with_color(frame, color)

        return video_CTHW
    
    def __call__(self, *args, **kwds):
        return self.augment_video(*args, **kwds)
    
def _plot(x: torch.Tensor):
    import matplotlib.pyplot as plt
    from einops import rearrange
    import numpy as np
    print(x.min(), x.max())
    x = rearrange(x, "c h w -> h w c")
    x = x.numpy()
    if x.max() < 1.: x *= 255
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(x)
    plt.savefig('test.png')


if __name__ == "__main__":
    train = LabelwiseColorFilterAugmentation('train', labels=list(range(20)), debug_file_prefix='debug_')
    distractor = LabelwiseColorFilterAugmentation('distracted', labels=list(range(20)), debug_file_prefix='debug_')
    eval = LabelwiseColorFilterAugmentation('eval', labels=list(range(20)), debug_file_prefix='debug_')

# from matplotlib.patches import Circle
# from multiprocessing import Pool


# def create_moving_dot_video(frames=16, height=128, width=128):
#     """Create a video of a moving white dot."""
#     video = torch.zeros((3, frames, height, width))
    
#     # Create circular motion
#     t = np.linspace(0, 2*np.pi, frames)
#     center_x, center_y = width//2, height//2
#     radius = min(width, height)//4
    
#     x = center_x + radius * np.cos(t)
#     y = center_y + radius * np.sin(t)
    
#     # Draw dots
#     for i in range(frames):
#         # Create gaussian blob
#         xx, yy = np.mgrid[0:height, 0:width]
#         circle = np.exp(-((xx - y[i])**2 + (yy - x[i])**2) / 100)
#         video[:, i] = torch.tensor(circle).unsqueeze(0).repeat(3, 1, 1)
    
#     return video

# def visualize_color_wheel(augmentation, labels):
#     """Visualize how colors are distributed on the color wheel."""
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_aspect('equal')
    
#     # Draw main circle
#     circle = plt.Circle((0, 0), 1, fill=False)
#     ax.add_artist(circle)
    
#     # Plot each label's color
#     for label in labels:
#         color = augmentation.assign_to_color(label)
#         hue = augmentation.labels_to_hues[label]
        
#         # Convert angle to x,y
#         angle = np.deg2rad(hue)
#         x = np.cos(angle)
#         y = np.sin(angle)
        
#         # Draw dot and label
#         ax.add_patch(Circle((x, y), 0.1, color=color))
#         ax.text(x*1.2, y*1.2, label, ha='center', va='center')
    
#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     plt.title("Color Distribution")
#     plt.grid(True)
#     return fig

# def process_with_workers(num_workers=4):
#     """Demonstrate color consistency across workers."""
#     labels = ["push", "pull", "lift", "drop", "slide"]
    
#     def worker_fn(worker_id):
#         augmentation = LabelwiseColorFilterAugmentation(split="train")
#         colors = {label: augmentation.assign_to_color(label) for label in labels}
#         return worker_id, colors
    
#     with Pool(num_workers) as pool:
#         results = pool.map(worker_fn, range(num_workers))
    
#     # Check if all workers got the same colors
#     colors_match = all(
#         results[0][1] == worker_colors 
#         for _, worker_colors in results[1:]
#     )
    
#     return results, colors_match

# if __name__ == "__main__":
#     from einops import rearrange
#     # Create test video
#     video = create_moving_dot_video()
#     video = rearrange(video, "C T H W -> T C H W")
    
#     # Create augmentation
#     augmentation = LabelwiseColorFilterAugmentation(split="train", alpha=0.5)
    
#     # Test labels
#     labels = ["push", "pull", "lift", "drop", "slide"]
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle("Label-wise Color Filter Visualization")
    
#     # Plot color wheel
#     wheel_fig = visualize_color_wheel(augmentation, labels)
    
#     # Plot original and filtered frames for each label
#     original_frame = video[:, 0].permute(1, 2, 0)
    
#     for idx, label in enumerate(labels):
#         row = idx // 3
#         col = idx % 3
#         ax = axes[row, col]
        
#         # Apply color filter
#         filtered_video = augmentation.augment_video(video.clone(), label)
#         filtered_video = rearrange(filtered_video, "T C H W -> T H W C")
#         filtered_frame = filtered_video[0, :]
#         # Plot filtered frame
#         ax.imshow(filtered_frame)
#         ax.set_title(f"Label: {label}")
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()
    
#     # Test worker consistency
#     results, colors_match = process_with_workers()
#     print(f"\nColor consistency across workers: {'✓' if colors_match else '✗'}")
    
#     # Print color assignments
#     print("\nColor assignments (RGB):")
#     for label in labels:
#         color = augmentation.assign_to_color(label)
#         print(f"{label:>10}: {color}")