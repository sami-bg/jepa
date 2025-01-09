from __future__ import annotations
import torch
from multiprocessing import Manager
import logging
from functools import cache

logger = logging.getLogger()

@cache
def warn_once(x: str): logger.warning(x)


class LabelwiseTemporalFlip:
    def __init__(self, split="train", flip: bool = False, randomize: bool = False):
        """
        Args:
            split: Data split ("train", "test", "val", "eval")
            flip: If True, reverse the sequence
            randomize: If True, randomly permute the sequence
            
        Note: flip and randomize cannot both be True
        """
        self.split = split
        assert self.split in {"train", "test", "val", "eval"}
        
        # XOR check - only one can be True
        assert not (flip and randomize), "Cannot both flip and randomize - choose one"
        assert flip or randomize, "Must choose either flip or randomize"
        self.flip = flip 
        self.randomize = randomize

        # Shared dictionary across workers
        manager = Manager()
        self.labels_to_augment = manager.dict()

    def should_augment(self, label: str) -> bool:
        if label in self.labels_to_augment:
            return self.labels_to_augment[label]

        # Same logic as before but for either operation
        num_labels = len(self.labels_to_augment)
        should_augment = (num_labels % 2) == (0 if self.split in {"train", "val"} else 1)
        
        self.labels_to_augment[label] = should_augment
        return should_augment

    def augment_video(self, video_TCHW: torch.Tensor, label: str) -> torch.Tensor:
        if label is None:
            warn_once(f'Received no label for LabelwiseTemporalFlip')
            return video_TCHW

        label = str(label)
        if self.should_augment(label):
            if self.flip:
                # Reverse the time dimension
                video_TCHW = torch.flip(video_TCHW, dims=[0])
            else:  # randomize
                # Randomly permute the time dimension
                T = video_TCHW.shape[0]
                perm = torch.randperm(T)
                video_TCHW = video_TCHW[perm]
            
        return video_TCHW
    

import torch
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

def create_number_sequence_video(frames=8, height=64, width=128):
    """Create a video where each frame shows its frame number prominently."""
    video = torch.zeros((3, frames, height, width))
    
    # Create a sequence of frames with numbers
    for i in range(frames):
        # Create a base frame
        frame = torch.zeros((height, width))
        
        # Add frame number text
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(width/20, height/20))
        plt.text(0.5, 0.5, str(i), fontsize=50, ha='center', va='center')
        plt.axis('off')
        
        # Convert plot to tensor
        fig.canvas.draw()
        plt_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_frame = plt_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_frame = torch.from_numpy(plt_frame).float() / 255.0
        plt_frame = rearrange(plt_frame, 'h w c -> c h w')
        
        # Add to video tensor
        video[:, i] = plt_frame
        
        plt.close(fig)
    
    return video


def make_grid(video_TCHW: torch.Tensor) -> torch.Tensor:
    """Convert video tensor to a grid of frames for visualization."""
    # Normalize to 0-1 if needed
    if video_TCHW.max() > 1:
        video_TCHW = video_TCHW / 255.0
    
    T, C, H, W = video_TCHW.shape
    grid = torch.zeros((H, T*W, C))
    
    for t in range(T):
        grid[:, t*W:(t+1)*W, :] = rearrange(video_TCHW[t], 'c h w -> h w c')
    
    return grid
def create_number_sequence_video(frames=8, height=64, width=128):
    """Create a video where each frame shows its frame number prominently."""
    video = torch.zeros((3, frames, height, width))
    
    # Create a sequence of frames with numbers
    for i in range(frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use cv2 to draw text instead of matplotlib
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
        textsize = cv2.getTextSize(text, font, 2, 3)[0]
        
        # Get coords to center text
        textX = (width - textsize[0]) // 2
        textY = (height + textsize[1]) // 2
        
        # Add text to image
        frame = cv2.putText(frame, text, (textX, textY), font, 2, (255, 255, 255), 3)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = rearrange(frame_tensor, 'h w c -> c h w')
        
        # Add to video tensor
        video[:, i] = frame_tensor
    
    return video

if __name__ == "__main__":
    # Create test video
    video = create_number_sequence_video(frames=8)
    video = rearrange(video, 'c t h w -> t c h w')
    
    # Create flip augmentation
    flip_aug = LabelwiseTemporalFlip(split="train", randomize=True)
    
    # Test labels
    labels = ["push", "pull", "lift", "drop"]
    
    # Create visualization
    fig, axes = plt.subplots(len(labels), 2, figsize=(15, 5*len(labels)))
    fig.suptitle("Label-wise Temporal Flip Visualization")
    
    for idx, label in enumerate(labels):
        # Get original and flipped sequences
        orig_video = video.clone()
        flipped_video = flip_aug.augment_video(video.clone(), label)
        
        # Show original sequence
        axes[idx, 0].imshow(make_grid(orig_video))
        axes[idx, 0].set_title(f"Label: {label} (Original)")
        axes[idx, 0].axis('off')
        
        # Show flipped sequence
        axes[idx, 1].imshow(make_grid(flipped_video))
        axes[idx, 1].set_title(f"Label: {label} ({'Flipped' if flip_aug.should_augment(label) else 'Not Flipped'})")
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print which labels got flipped
    for label in labels:
        flipped = flip_aug.should_augment(label)
        print(f"Label {label}: {'FLIPPED' if flipped else 'normal'}")
