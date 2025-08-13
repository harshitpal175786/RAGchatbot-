import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("M1 GPU (MPS) is available!")
    x = torch.ones(1, device=device)
    print(x)
else:
    print("MPS device not found.")
