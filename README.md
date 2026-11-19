# pijepa

P-IJEPA trains an image encoder to predict features for hidden image patches. Its additional loss helps preserve similarity between patches. [Research paper](https://doi.org/10.1016/j.patrec.2026.07.005)

## Run on one GPU

1. Open a terminal in this repository. Use Linux, Python 3.11, and an NVIDIA GPU that supports BF16 and CUDA 11.8.
2. Create an environment and install the dependencies.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
   python -m pip install -r requirements.txt
   ```

3. Put ImageNet-100 images in the following folders. Use the same class names in both sets.

   ```text
   datasets/imagenet100/train/<class>/<image>
   datasets/imagenet100/val/<class>/<image>
   ```

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

This implementation builds on [I-JEPA](https://github.com/facebookresearch/ijepa). See the [license](LICENSE) for terms.
