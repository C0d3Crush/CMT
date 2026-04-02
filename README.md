![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# CMT

Keunsoo Ko and Chang-Su Kim

Official PyTorch Code for "Continuously Masked Transformer for Image Inpainting, ICCV, 2023"

### Installation
Download repository:
```
    $ git clone https://github.com/keunsoo-ko/CMT.git
```
Download [pre-trained model on Places2](https://drive.google.com/file/d/1zLkKixPnuoAY1k4fdq6JjidlRafKMhXN/view?usp=sharing) or [pre-trained model on CelebA](https://drive.google.com/file/d/1e6EbwGnMGgGXAn4QLffT_Zx_BbidBSbR/view?usp=sharing)

### Usage
Run Test:
```
    $ python demo.py --ckpt CMT.pth(put downloaded model path) --img_path ./samples/test_img --mask_path ./samples/test_mask --output_path ./samples/results
```

### Device support (added in this fork)

You can specify whether to run on CPU or GPU:

```
    $ python demo.py --ckpt CMT.pth --img_path ./samples/test_img --mask_path ./samples/test_mask --output_path ./samples/results --device cpu
```

GPU:

```
    $ python demo.py --ckpt CMT.pth --img_path ./samples/test_img --mask_path ./samples/test_mask --output_path ./samples/results --device cuda
```

The checkpoint loader uses `map_location`, therefore models trained on GPU can also be loaded on CPU without modification.

