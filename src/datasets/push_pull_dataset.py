import torch
import random
from torch.utils.data import Dataset
from typing import NamedTuple, Literal

class Sample(NamedTuple):
    frames_BTCHW: torch.Tensor
    labels_BTL: torch.Tensor


class PushPullDataset(Dataset):
    def __init__(
            self,
            num_datapoints,
            batch_size,
            timesteps,
            num_channels,
            height,
            width,
            device = torch.device("cpu")
        ):
        
        super().__init__()
        self.num_datapoints: int   = num_datapoints
        self.batch_size: int    = batch_size
        self.timesteps: int     = timesteps
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
        """
        Frame dimensions: self.height, self.width.
        Video timesteps: self.timesteps
        Device: self.device

        Frame 0:
        1. Put a circle of a random color in a random position in the frame.
        2. Put a square of a random color in a random position in the frame, that does not cover the circle.
        For each frame 1->self.timesteps:
        3. If the direction is -1, move the square away from the circle.
        4. If the direction is 1,  move the square towards the circle.

        Conditions: the square must never be pushed fully outside of the boundaries of the image. It is okay if it is partially outside the images.
        """
        # Initialize empty video tensor
        video = torch.zeros(self.timesteps, self.num_channels, self.height, self.width, device=self.device)
        
        # Generate random colors for circle and square (RGB)
        circle_color = torch.rand(3, device=self.device)
        square_color = torch.rand(3, device=self.device)
        
        # Random circle position (center coordinates)
        circle_x = torch.randint(self.width//4, 3*self.width//4, (1,), device=self.device).item()
        circle_y = torch.randint(self.height//4, 3*self.height//4, (1,), device=self.device).item()
        circle_radius = min(self.height, self.width) // 10
        
        # Initial square position and size
        square_size = min(self.height, self.width) // 8
        
        # Place square at a random position that doesn't overlap with circle
        while True:
            square_x = torch.randint(square_size//2, self.width-square_size//2, (1,), device=self.device).item()
            square_y = torch.randint(square_size//2, self.height-square_size//2, (1,), device=self.device).item()
            
            # Check if square is far enough from circle
            dist = ((square_x - circle_x)**2 + (square_y - circle_y)**2)**0.5
            if dist > circle_radius + square_size:
                break
        
        # Calculate movement vector (normalized direction from square to circle)
        dx = circle_x - square_x
        dy = circle_y - square_y
        dist = max((dx**2 + dy**2)**0.5, 1e-6)  # avoid division by zero
        dx, dy = dx/dist, dy/dist
        
        # Movement speed
        speed = min(self.height, self.width) // 40
        
        # Generate frames
        for t in range(self.timesteps):
            # Draw circle
            y_grid, x_grid = torch.meshgrid(
                torch.arange(self.height, device=self.device),
                torch.arange(self.width, device=self.device),
                indexing='ij'
            )
            circle_mask = ((x_grid - circle_x)**2 + (y_grid - circle_y)**2 <= circle_radius**2)
            
            # Draw square
            square_mask = (
                (x_grid >= square_x - square_size//2) & 
                (x_grid < square_x + square_size//2) & 
                (y_grid >= square_y - square_size//2) & 
                (y_grid < square_y + square_size//2)
            )
            
            # Add shapes to frame with their colors
            for c in range(self.num_channels):
                video[t, c][circle_mask] = circle_color[c]
                video[t, c][square_mask] = square_color[c]
            
            # Update square position for next frame
            if t < self.timesteps - 1:
                # Move square towards/away from circle based on direction
                new_square_x = square_x + direction * speed * dx
                new_square_y = square_y + direction * speed * dy
                
                # Constrain square position to prevent it from leaving the frame entirely
                new_square_x = max(square_size//2, min(self.width - square_size//2, new_square_x))
                new_square_y = max(square_size//2, min(self.height - square_size//2, new_square_y))
                
                square_x, square_y = new_square_x, new_square_y
        
        return video
        
    def generate_multistep_batch(self) -> tuple[torch.Tensor, list[int]]:
        labels = self._random_direction(self.batch_size)
        samples_TCHW = [self.generate_multistep_sample(dir) for dir in labels]
        return torch.stack(samples_TCHW, dim=0), labels

    def __len__(self):
        return self.num_datapoints
    
    def __getitem__(self, i):
        return self.generate_multistep_batch()
    
    def __iter__(self):
        for _ in range(self.num_datapoints): yield self.generate_multistep_batch()
    

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
        timesteps=16,
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
    visualize_batch(filtered, save_path="tinted_video_batch.gif")
    
    filtered_test = torch.stack([
        aug_test.augment_video(batch_videos[i], labels[i])
        for i in range(batch_videos.shape[0])
    ], dim=0)

    visualize_batch(filtered_test, save_path="eval_tinted_video_batch.gif")
