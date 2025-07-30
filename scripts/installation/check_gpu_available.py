# check pytorch, tensorflow and jax
import torch


def check_gpu_available():
    gpu_available = torch.cuda.is_available()
    assert gpu_available, "GPU is not available for PyTorch."
    print("GPU is available.")


if __name__ == "__main__":
    check_gpu_available()