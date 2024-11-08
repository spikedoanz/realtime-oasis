import torch
from flash_attn import flash_attn_func
import time

def test_flash_attention():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    seq_len = 1024
    n_heads = 8
    head_dim = 64
    
    # Generate random query, key, value tensors
    # Shape: [batch_size, seq_len, n_heads, head_dim]
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Input tensor shapes: {q.shape}")
    
    try:
        # Warmup
        for _ in range(3):
            out = flash_attn_func(q, k, v)
        
        # Benchmark
        start_time = time.time()
        n_iterations = 100
        
        for _ in range(n_iterations):
            out = flash_attn_func(q, k, v)
            
        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_iterations * 1000  # Convert to milliseconds
        
        print("\nTest successful!")
        print(f"Average execution time: {avg_time:.2f} ms")
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        print(f"\nTest failed with error:\n{str(e)}")

if __name__ == "__main__":
    test_flash_attention()
