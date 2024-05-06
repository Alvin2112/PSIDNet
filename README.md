# [Progressive Stereo Image Dehazing Network via Cross-view Region Interaction](https://ieeexplore.ieee.org/document/10443871)
> **Abstract**: Stereo image dehazing aims to restore haze-free images by leveraging the complementary information contained in binocular images. Current methods primarily focus on designing image-level modules and pipelines to utilize complementary information between the left and right-view images. However, these image-level cross-view interactions overlook regional differences in haze concentration and stereo image disparity maps. Consequently, we propose a Progressive Stereo Image Dehazing Network via Cross-view Region Interaction, termed PSIDNet, which fully considers the internal characteristics and external manifestation of haze and disparity, and explicitly addresses the stereo image dehazing task by a regional-aware interactive mechanism. Specifically, we divide hazy images into regions and independently interact with left and right-view information at region levels, meaning weights are not shared across regional patches. This approach allows us to treat different regions with different priorities, i.e., concentrate on regional patches with heavier haze concentration and larger disparities, hence enabling more accurate restoration of hazy images. Furthermore, we introduce an effective cross-view region interactive block that extracts information based on the channel dimension of dual views and later adopts matrix multiplication to generate mutual attention maps based on the fused features. Extensive experiments on synthetic and real-scenario datasets demonstrate the efficacy of our method, compared to other related monocular and stereo image dehazing and restoration methods.

![](https://github.com/Alvin2112/PSIDNet/blob/main/fig/network.jpg)

## Training and Evaluation

### Train
You need to modify the training path in the `datasets.py` file to train your own dataset.
```
python train.py
```

### Test
You need to modify the testing path in the datasets.py file to test your own dataset.
```
python test.py

```

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{wang2024progressive,
  title={Progressive Stereo Image Dehazing Network via Cross-view Region Interaction},
  author={Wang, Junhu and Wei, Yanyan and Zhang, Zhao and Fan, Jicong and Zhao, Yang and Yang, Yi and Wang, Meng},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}

```
