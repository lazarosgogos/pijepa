# pijepa

P-IJEPA trains an image encoder to predict features for hidden image patches. Its additional loss helps preserve similarity between patches. [Research paper](https://doi.org/10.1016/j.patrec.2026.07.005)

## Run on one GPU

P-IJEPA originally used a Conda environment on Linux. Both setup options require an NVIDIA GPU that supports BF16 and CUDA 11.8.

1. Open a terminal in this repository.
2. Choose one setup option.

   **Conda (original setup)**  
   Create the environment with Python 3.12. Install PyTorch with Conda, then install the remaining packages with pip.

   ```bash
   conda create -n pijepa python=3.12 pip pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=11.8 -c pytorch -c nvidia -c defaults
   conda activate pijepa
   python -m pip install -r requirements.txt
   ```

   **Fallback with venv and pip**  
   If Conda is unavailable, create a virtual environment with Python 3.11.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
   python -m pip install -r requirements.txt
   ```

3. Create the ImageNet-100 folders under the repository root. Use the same class folder names in `train/` and `val/`.

   ```text
   datasets/imagenet100/
   ├── train/
   │   ├── class_a/image1.jpg
   │   └── class_b/image2.jpg
   └── val/
       ├── class_a/image3.jpg
       └── class_b/image4.jpg
   ```

   Replace `class_a` and `class_b` with your class names. The loader uses these names to assign labels.
   Put at least one supported image, such as a JPEG, in each class folder.

4. Start training.

   ```bash
   python main.py --fname configs/pijepa.yaml --devices cuda:0
   ```

Find checkpoints, training logs, and TensorBoard data in `runs/pijepa/`. Adjust `data.batch_size` to fit your GPU memory.

## Examples

| Configuration | Purpose |
| --- | --- |
| `configs/pijepa.yaml` | Train ViT-B/16 with the combined loss (`L2_PKT`). |
| `configs/ijepa.yaml` | Train the same model with the baseline loss (`L2`). |
| `configs/probe.yaml` | Train a linear classifier on features from saved P-IJEPA checkpoints. |

For the baseline, use `configs/ijepa.yaml` in step 4. For linear probing, wait for training to save a numbered checkpoint. Then run:

```bash
python pmulti-linear-probing.py --fname configs/probe.yaml
```

Notice the absence of the `--devices ` command line argument in the evaluation bash script. It is defined inside the config itself by design.  

This implementation builds on [I-JEPA](https://github.com/facebookresearch/ijepa). See the [license](LICENSE) for terms.
