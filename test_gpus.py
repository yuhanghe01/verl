import torch

def test_amd_multigpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Ensure that ROCm is installed and PyTorch is built with ROCm support.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of AMD GPUs detected: {num_gpus}")
    
    if num_gpus < 2:
        print("Less than 2 GPUs detected. Ensure multiple AMD GPUs are available.")
        return

    try:
        # Create a tensor and distribute across multiple GPUs
        tensors = [torch.ones(10, device=f'cuda:{i}') for i in range(num_gpus)]
        
        # Perform basic computation
        results = [t * 2 for t in tensors]
        
        # Verify computation
        for i, res in enumerate(results):
            if not torch.all(res == 2):
                print(f"Computation failed on GPU {i}")
                return

        print("All AMD GPUs are working correctly.")
    
    except Exception as e:
        print(f"Error encountered during multi-GPU test: {e}")

if __name__ == "__main__":
    test_amd_multigpu()
