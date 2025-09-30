import torch

# replace with your actual path
checkpoint_path = "multi_task_bert.pth"

# load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# check what type it is
print("Type of checkpoint:", type(checkpoint))

# if it's a dict, print keys
if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
else:
    print("Checkpoint is not a dict. It might be the full model or something else.")
