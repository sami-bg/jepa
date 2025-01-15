import torch
from functools import partial
from einops import rearrange
import torch.utils.data.distributed

from src.datasets.utils.moving_dot.single import ContinuousMotionDataset, Sample


def make_movingdot_dataset(
    batch_size,
    n_steps=16,
    noise=0.0,
    static_noise=0.0,
    structured_noise=False,
    transform=None,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    pin_mem=True,
    collator=None
):
    dataset = ContinuousMotionDataset(
        size=1_000_000,  # Pre-training size from paper
        batch_size=batch_size, 
        n_steps=n_steps,
        noise=noise,
        static_noise=static_noise,
        structured_noise=structured_noise,
        img_size=224,
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
        collate_fn = partial(dot_collate_fn, frames_per_clip=n_steps)

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

def dot_collate_fn(samples: list[Sample] , frames_per_clip: int):
    """Match VideoDataset's format:
    - buffer: list of clips, each clip containing frames
    - label: dummy label since we have no classes
    - clip_indices: temporal indices for frames in each clip
    """
    sample = samples[0]  # Get single Sample since batch_size=1
    states = sample.states  # [batch_size, T, 1, 28, 28]
    actions = sample.actions
    locations = sample.locations
    num_clips = states.shape[1] // frames_per_clip

    # First collect ALL indices
    clip_indices = []
    for i in range(num_clips):
        indices = torch.arange(i*frames_per_clip, (i+1)*frames_per_clip)
        clip_indices.append(indices)

    # un-grayscale 
    states = states.repeat(1, 1, 3, 1, 1)
    
    # Reorder dimensions to [batch_size, channels, frames, height, width]
    # Return in format expected by V-JEPA training loop
    # ie udata, maskenc, maskpred
    # udata[0] = [.]
    # udata[0][0] = 6,3,16,384,384
    states = rearrange(states, "b f c h w -> b c f h w")
    
    return states, actions.squeeze(2), locations.squeeze(2), clip_indices

def combined_collate_fn(samples, mask_collator, frames_per_clip):
    """
    Combines dot dataset collation with V-JEPA mask generation
    
    Args:
        samples: List of Sample objects from ContinuousMotionDataset
        mask_collator: V-JEPA's MaskCollator instance
        frames_per_clip: Number of frames per clip
    """
    # First do dot dataset collation 
    states, actions, labels, clip_indices = dot_collate_fn(samples, frames_per_clip)
    
    # Create batch tuple as expected by MaskCollator
    # MaskCollator expects a batch that can be processed by default_collate
    batch = [(state,) for state in states]  # Make each state a tuple
    
    # Get masks using V-JEPA's collator
    # This returns (collated_batch, collated_masks_enc, collated_masks_pred)
    _, masks_enc, masks_pred = mask_collator(batch)

    # NOTE usually list is done from videodataset transform buffer = [self.transform(clip) for clip in buffer]
    # btu we dont have that datset here
    return ([states], actions, labels, clip_indices), masks_enc, masks_pred


def visualize_sample(states, actions):
    """
    Visualize the dot movement and corresponding actions
    states: [T, 28, 28] numpy array
    actions: [T, 2] numpy array
    """
    import matplotlib.animation as animation
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(frame):
        # Clear previous frame
        ax1.clear()
        ax2.clear()
        
        # Plot current frame
        ax1.imshow(states[frame], cmap='gray')
        ax1.set_title(f'Frame {frame}')
        
        # Plot action vector
        if frame < len(actions):
            ax2.quiver(0, 0, actions[frame, 0], actions[frame, 1], 
                      angles='xy', scale_units='xy', scale=1)
            ax2.set_xlim(-0.2, 0.2)
            ax2.set_ylim(-0.2, 0.2)
            ax2.set_title(f'Action Vector\n({actions[frame, 0]:.3f}, {actions[frame, 1]:.3f})')
            ax2.grid(True)
            
    ani = animation.FuncAnimation(fig, update, frames=len(states), 
                                interval=200, repeat=True)
    plt.show()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    ds, dl, dsts = make_movingdot_dataset(16, n_steps=8, num_workers=0, noise=0.5)
    ids = iter(dl)
    while x:=next(ids):
        states, actions, clip_indices = x
        visualize_sample(states[0], actions[0])