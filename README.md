# Mobile-PolypNet : Light-weight Colon Polyp Segmentation Network for Low Resources Settings

This is an unofficial PyTorch implementation of [Mobile-PolypNet](https://github.com/rkarmaka/Mobile-PolypNet).


## Requirements

```
pip install -r requirements.txt
```


## Model


<div align="center">

<img src="https://raw.githubusercontent.com/rkarmaka/Mobile-PolypNet/main/figs/model_arch_mod.png" width="600">

</div>

Mobile-PolypNet model backbone architecture with the bottleneck residual blocks and skip connection where x, e and c represents the number of bottleneck residual block, number of filters for expansion operation, and number of filters for contraction respectively.
