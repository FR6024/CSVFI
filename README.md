# CSVFI: Capture More, Synthesize Better: Video Frame Interpolation with Larger Receptive Field and Structural Priors

This repo is the official implementation of *Capture More, Synthesize Better: Video Frame Interpolation with Larger Receptive Field and Structural Priors*, IEEE TCSVT 2026.

## Toy Demo
:heart: **If the playback is lagging, please wait for a moment to buffer, or you can directly watch the original file in the *GIF* folder.**

:dog2:

![  ](GIFS/dog.gif)

:dog: 


![](GIFS/libby.gif)

:car:

![](GIFS/draft.gif)

## Dependencies
Create conda env:
```
conda env create -f environment.yml
```


## Train
* Download the [Vimeo-90K septuplets](http://toflow.csail.mit.edu/) dataset. 
* Then train CSVFI (DDP in linux):

```
# CSVFI-T
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=6027 main-DDP.py --model=CSVFI_T --data_root=<dataset_path> --batch_=5
# CSVFI-S
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=6027 main-DDP.py --model=CSVFI_S --data_root=<dataset_path> --batch_=5
# CSVFI-B
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=6027 main-DDP.py --model=CSVFI_B --data_root=<dataset_path> --batch_=5
```

## Test
After training, you can evaluate the model with following command:

```
# CSVFI-T
python test.py --model=CSVFI_T --dataset vimeo90K_septuplet --data_root=<dataset_path> --load_from checkpoints/model_best_T.pth
# CSVFI-S
python test.py --model=CSVFI_S --dataset vimeo90K_septuplet --data_root=<dataset_path> --load_from checkpoints/model_best_S.pth
# CSVFI-B
python test.py --model=CSVFI_B --dataset vimeo90K_septuplet --data_root=<dataset_path> --load_from checkpoints/model_best_B.pth
```


Please consider citing this paper if you find the code and data useful in your research:
```

@article{zhou2025dual,
  title={Dual-Guided Video Frame Interpolation with Spatial-Temporal Global Attention},
  author={Zhou, Baojun and Huang, Xinpeng and Li, Gongyang and Yang, Chao and Shen, Liquan and An, Ping},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}

@article{zhou2026capture,
  title={Capture More, Synthesize Better: Video Frame Interpolation with Larger Receptive Field and Structural Priors},
  author={Zhou, Baojun and Huang, Xinpeng and Chen, Jieyu and Kaaniche, Mounir and An, Ping},
  journal={IEEE IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}

```


## Acknowledgment
Some references to the code, many thanks for their excellent work:
* FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation, arXiv 2021 [Code](https://github.com/tarun005/FLAVR)
* Video Frame Interpolation Transformer, CVPR 2022 [Code](https://https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer)
* Video Frame Interpolation via Adaptive Separable Convolution, ICCV 2017 [Code](https://https://github.com/sniklaus/sepconv-slomo)
