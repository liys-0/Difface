# Mock Contrastive Learning for Difface

This script isolates the InfoNCE contrastive learning logic from the main Difface repository. It focuses entirely on the alignment phase between DNA sequences and 3D face representations. You can use it to understand the core training loop without dealing with the full pipeline.

### Data Mocking

The full Difface project relies on heavy SpiralConv and Transformer architectures. This mock script replaces those complex networks with simple MLPs. We also generate random tensors to simulate the incoming batches of DNA and Face data. Such an approach keeps the code lightweight and easy to run on any machine.

### How to Run

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install PyTorch: `pip install torch`
4. Run the script: `python mock_contrastive.py`

### Expected Output

The script prints its training progress directly to the console. You'll see the InfoNCE contrastive loss dropping steadily as the epochs progress. This decreasing loss confirms that the mock models are successfully learning to align the paired DNA and face embeddings in the shared latent space.
