import torch
import logging

logging.basicConfig(level=logging.INFO)

def check_cuda():
    logging.info(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"Device name: {torch.cuda.get_device_name()}")
        logging.info(f"Device count: {torch.cuda.device_count()}")
        
        # Test CUDA with a small tensor operation
        logging.info("\nTesting CUDA with tensor operation:")
        x = torch.rand(5, 3)
        if torch.cuda.is_available():
            x = x.cuda()
            logging.info("Successfully created tensor on CUDA")
            logging.info(f"Tensor device: {x.device}")
    else:
        logging.info("CUDA is not available on this system")
        logging.info("Make sure you have:")
        logging.info("1. Compatible NVIDIA GPU")
        logging.info("2. NVIDIA drivers installed")
        logging.info("3. CUDA toolkit installed")
        logging.info("4. PyTorch with CUDA support installed")

if __name__ == "__main__":
    check_cuda()
