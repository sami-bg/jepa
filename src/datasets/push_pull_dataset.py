import torch
import random
from torch.utils.data import Dataset
from typing import NamedTuple, Literal
from functools import partial
from einops import rearrange

class Sample(NamedTuple):
    frames_BTHWC: torch.Tensor
    labels_B: torch.Tensor


class PushPullDataset(Dataset):
    def __init__(
            self,
            num_datapoints,
            batch_size,
            timesteps,
            num_channels,
            height,
            width,
            transform=None,
            device = torch.device("cpu"),
            split='train'
        ):
        
        super().__init__()
        self.split                 = split
        self.num_datapoints: int   = num_datapoints
        self.batch_size: int    = batch_size
        self.timesteps: int     = timesteps
        self.transform          = transform
        self.num_channels: int  = num_channels
        self.height: int        = height
        self.width: int         = width
        self.device             = device

    DIRECTIONS = {
        'AWAY': -1,
        'TOWARDS': 1
    }

    def _random_direction(self, batch_size: int) -> list[Literal[1, -1]]:
        directions = list(self.DIRECTIONS.values())
        counts = [1, 1]
        probabilities = [c / sum(counts) for c in counts]
        
        return random.choices(
            population=directions,
            weights=probabilities,
            k=batch_size
        )

    def generate_multistep_sample(self, direction: Literal[1, -1]) -> torch.Tensor:
        # Initialize empty video tensor
        video = torch.zeros(self.timesteps, self.num_channels, self.height, self.width, device=self.device)
        
        # Generate random colors for circle and square (RGB)
        circle_color = torch.rand(3, device=self.device)
        square_color = torch.rand(3, device=self.device)
        
        # Place circle in center
        circle_x = self.width // 2
        circle_y = self.height // 2
        # Made circle smaller (changed from //10 to //15)
        circle_radius = min(self.height, self.width) // 15
        
        # Initial square position and size - made square smaller (changed from //8 to //12)
        square_size = min(self.height, self.width) // 12
        
        # Calculate radius where square should be placed to be equidistant
        # from circle and frame edge
        max_radius = min(self.height, self.width) // 2  
        diagonal_radius = ((self.height//2)**2 + (self.width//2)**2)**0.5  
        placement_radius = (max_radius + diagonal_radius) / 4  
        
        # Random angle for square placement
        angle = torch.rand(1, device=self.device).item() * 2 * torch.pi
        
        # Place square at this radius and angle
        square_x = circle_x + placement_radius * torch.cos(torch.tensor(angle))
        square_y = circle_y + placement_radius * torch.sin(torch.tensor(angle))
        
        # Calculate movement vector (normalized direction from square to circle)
        dx = circle_x - square_x
        dy = circle_y - square_y
        dist = max((dx**2 + dy**2)**0.5, 1e-6)  # avoid division by zero
        dx, dy = dx/dist, dy/dist
        
        # Movement speed - made slower (changed from //40 to //80)
        speed = min(self.height, self.width) / 240
        
        # Rest of the function remains the same...
        for t in range(self.timesteps):
            y_grid, x_grid = torch.meshgrid(
                torch.arange(self.height, device=self.device),
                torch.arange(self.width, device=self.device),
                indexing='ij'
            )
            circle_mask = ((x_grid - circle_x)**2 + (y_grid - circle_y)**2 <= circle_radius**2)
            
            square_mask = (
                (x_grid >= square_x - square_size//2) & 
                (x_grid < square_x + square_size//2) & 
                (y_grid >= square_y - square_size//2) & 
                (y_grid < square_y + square_size//2)
            )
            
            for c in range(self.num_channels):
                video[t, c][circle_mask] = circle_color[c]
                video[t, c][square_mask] = square_color[c]
            
            if t < self.timesteps - 1:
                new_square_x = square_x + direction * speed * dx
                new_square_y = square_y + direction * speed * dy
                
                new_square_x = max(square_size//2, min(self.width - square_size//2, new_square_x))
                new_square_y = max(square_size//2, min(self.height - square_size//2, new_square_y))
                
                square_x, square_y = new_square_x, new_square_y
        
        return rearrange(video, "t c h w -> t h w c")

    def generate_multistep_batch(self) -> Sample:
        labels_B = torch.tensor(self._random_direction(self.batch_size))
        # this is what jepa needs i guess
        frames_BTHWC = torch.stack([self.generate_multistep_sample(dir) for dir in labels_B], dim=0)

        if self.transform:
            frames_BTHWC = torch.stack([self.transform(clip, label) for clip, label in zip(frames_BTHWC, labels_B)], dim=0)


        sample = Sample(
            frames_BTHWC=frames_BTHWC,
            labels_B=labels_B
        )
        return sample

    def __len__(self):
        return self.num_datapoints
    
    def __getitem__(self, i):
        return self.generate_multistep_batch()
    
    def __iter__(self):
        for _ in range(self.num_datapoints): yield self.generate_multistep_batch()


def make_pushpull_dataset(
    batch_size,
    n_steps=16,
    transform=None,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    pin_mem=True,
    collator=None,
    split='train',
):
    dataset = PushPullDataset(
        num_datapoints=1_000_000,  # Pre-training size from paper
        batch_size=batch_size, 
        timesteps=n_steps,
        transform=transform,
        num_channels=3,
        height=224,
        width=224,
        split=split,
        device=torch.device("cpu"),  # Move to GPU in transform
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    if collator:
        collate_fn = lambda x: combined_collate_fn(x, collator, n_steps)
    else:
        collate_fn = partial(pushpull_collate_fn, frames_per_clip=n_steps)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # NOTE dataset does batching itself
        sampler=dist_sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_mem,
        collate_fn=collate_fn
    )

    return dataset, data_loader, dist_sampler


def pushpull_collate_fn(samples: list[Sample] , frames_per_clip: int):
    """Match VideoDataset's format:
    - buffer: list of clips, each clip containing frames
    - label: dummy label since we have no classes
    - clip_indices: temporal indices for frames in each clip
    """
    sample = samples[0]  # Get single Sample since batch_size=1
    states = sample.frames_BTHWC  # [batch_size, T, 1, 28, 28]
    labels = sample.labels_B
    
    num_clips = states.shape[1] // frames_per_clip

    # First collect ALL indices
    clip_indices = []
    for i in range(num_clips):
        indices = torch.arange(i*frames_per_clip, (i+1)*frames_per_clip)
        clip_indices.append(indices)
    
    return states, labels, clip_indices


def combined_collate_fn(samples, mask_collator, frames_per_clip):
    """
    Combines dot dataset collation with V-JEPA mask generation
    
    Args:
        samples: List of Sample objects from ContinuousMotionDataset
        mask_collator: V-JEPA's MaskCollator instance
        frames_per_clip: Number of frames per clip
    """
    # First do dot dataset collation 
    states, labels, clip_indices = pushpull_collate_fn(samples, frames_per_clip)
    
    # Create batch tuple as expected by MaskCollator
    # MaskCollator expects a batch that can be processed by default_collate
    batch = [(state,) for state in states]  # Make each state a tuple
    
    # Get masks using V-JEPA's collator
    # This returns (collated_batch, collated_masks_enc, collated_masks_pred)
    _, masks_enc, masks_pred = mask_collator(batch)

    # NOTE usually list is done from videodataset transform buffer = [self.transform(clip) for clip in buffer]
    # btu we dont have that datset here
    return ([states], labels, clip_indices), masks_enc, masks_pred



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize_single_video(video_tensor: torch.Tensor, save_path: str = None):
    """
    Visualize a single video tensor with shape (T, C, H, W)
    
    Args:
        video_tensor: Tensor of shape (T, C, H, W)
        save_path: Optional path to save the animation as a gif
    """
    # Move to CPU and convert to numpy
    video = video_tensor.cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create initial plot
    im = ax.imshow(np.transpose(video[0], (1, 2, 0)))
    
    def update(frame):
        im.set_array(np.transpose(video[frame], (1, 2, 0)))
        return [im]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(video), 
        interval=100, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    plt.show()

def visualize_batch(batch_tensor: torch.Tensor, max_videos: int = 4, save_path: str = None):
    """
    Visualize a batch of videos with shape (B, T, C, H, W)
    
    Args:
        batch_tensor: Tensor of shape (B, T, C, H, W)
        max_videos: Maximum number of videos to display
        save_path: Optional path to save the animation as a gif
    """
    # Move to CPU and convert to numpy
    batch = batch_tensor.cpu().numpy()
    
    # Limit number of videos to display
    n_videos = min(batch.shape[0], max_videos)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_videos, figsize=(4*n_videos, 4))
    if n_videos == 1:
        axes = [axes]
    
    # Remove ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Create initial plots
    ims = [ax.imshow(np.transpose(batch[i, 0], (1, 2, 0))) 
           for i, ax in enumerate(axes)]
    
    def update(frame):
        for i, im in enumerate(ims):
            im.set_array(np.transpose(batch[i, frame], (1, 2, 0)))
        return ims
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=batch.shape[1], 
        interval=100, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = PushPullDataset(
        num_datapoints=100,
        batch_size=4,
        timesteps=32,
        num_channels=3,
        height=64,
        width=64
    )
    
    # Get a batch
    batch_videos, labels = dataset[0]
    
    # Visualize single video from batch
    print("Visualizing single video...")
    # visualize_single_video(batch_videos[0], save_path="single_video.gif")
    
    # Visualize batch
    print("Visualizing batch...")
    # visualize_batch(batch_videos, save_path="batch_videos.gif")

    from src.datasets.utils.video.labelwise_color_filter_augmentation import LabelwiseColorFilterAugmentation
    aug_train = LabelwiseColorFilterAugmentation(split="train", alpha=0.3)
    aug_test = LabelwiseColorFilterAugmentation(split="test", alpha=0.3)

    filtered = torch.stack([
        aug_train.augment_video(batch_videos[i], labels[i])
        for i in range(batch_videos.shape[0])
    ], dim=0)
    visualize_batch(filtered, save_path="tinted_video_batch2.gif")
    
    filtered_test = torch.stack([
        aug_test.augment_video(batch_videos[i], labels[i])
        for i in range(batch_videos.shape[0])
    ], dim=0)

    visualize_batch(filtered_test, save_path="eval_tinted_video_batch2.gif")
