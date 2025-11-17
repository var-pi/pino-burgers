import torch
from pathlib import Path

root_dir = Path('data')  # Adjust if needed, e.g., Path('/Users/var-pi/Repos/pino-burdgers/data')

# For train file
train_file = root_dir / 'burgers_train_16.pt'
if train_file.exists():
    data = torch.load(train_file)
    n_train = data['x'].shape[0]
    print(f"Train samples: {n_train}")
    print(f"Train x shape: {data['x'].shape}")  # e.g., (1000, 1, 16)
    print(f"Train y shape: {data['y'].shape}")  # e.g., (1000, 1, 26, 16)  # +1 for t=0
else:
    print(f"Train file not found: {train_file}")

# For test file
test_file = root_dir / 'burgers_test_16.pt'
if test_file.exists():
    data = torch.load(test_file)
    n_test = data['x'].shape[0]
    print(f"Test samples: {n_test}")
    print(f"Test x shape: {data['x'].shape}")
    print(f"Test y shape: {data['y'].shape}")
else:
    print(f"Test file not found: {test_file}")
