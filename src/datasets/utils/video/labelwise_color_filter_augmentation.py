from __future__ import annotations
import logging
import torch
from toolz import sliding_window
from functools import cache
from multiprocessing import Manager

import colorsys

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

    def __init__(self, split="train", alpha: float = 0.3, normalize_fn = None):
        # NOTE This should be consistent across all dataloader processes.
        self.split = split
        assert self.split in {"train", "test", "val", "eval"}
        self.alpha = alpha
        assert 0 <= self.alpha <= 1.
        self.normalize_fn = normalize_fn


        if self.split in {"train", "val"}:
            self.start_hue = 0
        elif self.split in {"test", "eval"}:
            self.start_hue = 180

        # NOTE This will break with multi-node training (e.g. slurm on more than 1 node) because
        # we would need to sync across nodes and not just across processes on 1 machine. To fix
        # this, you would need to get all labels up front and compute the mapping deterministically.
        # When it comes time to do so, we will likely be training on a large and established dataset
        # so it won't be an issue. 
        manager = Manager()
        self.labels_to_hues = manager.dict()
        self.labels_to_color = manager.dict()


    def assign_to_color(self, label: str) -> COLOR:
        # NOTE Each time a new label is added, add it maximally-between all the colors that already exist
        if label in self.labels_to_color:
            return self.labels_to_color[label]
        
        if (num_colors := len(self.labels_to_color)) == 0:
            hue = self.start_hue
        elif num_colors == 1:
            hue = (self.start_hue + 180) % 360
        else:
            exiting_hues = list(self.labels_to_hues.values())
            hue1, hue2 = max_gap_elts_radial(exiting_hues)
            print(f'{hue1=} {hue2=}')
            hue = (hue1 + hue2) / 2
            hue %= 360
        
        # NOTE Hue needs to be between 0 and 1
        self.labels_to_hues[label] = hue
        self.labels_to_color[label] = colorsys.hsv_to_rgb(hue / 360, s=1., v=1.)
        return self.labels_to_color[label]

    def _augment_frame_with_color(self, frame_CHW: torch.Tensor, color: torch.Tensor) -> torch.Tensor:

        tinted_frame = (1 - self.alpha) * frame_CHW + self.alpha * color        
        return torch.clamp(tinted_frame, 0, 1)
    

    def augment_video(self, video_CTHW: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if label is None:
            warn_once(f'Received no label for LabelwiseColorFilterAugmentation')
            return video_CTHW
        
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