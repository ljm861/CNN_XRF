# CNN_XRF
2D CNN model for X-ray Fluorescence (XRF) imaging of gold nanoparticles

## Requirements
* Python (>3.5)
* Pytorch (>1.0)

## Run
* `Input = torch.randn(8, 1, 20, 20)` (Size of input images : 20 x 20)

```bash
python cnn2d.py
```

> output shape : torch.Size([8, 1, 20, 20])
