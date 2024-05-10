# GL-MSDA

Official implementation of ICRA2024 paper "Sim-to-Real Grasp Detection with Global-to-Local RGB-D Adaptation"

## Installation

The implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [DGCAN](https://github.com/mahaoxiang822/dgcan).

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Dataset

To prepare the dataset,

1. download the [Graspnet-1billion](https://graspnet.net/index.html).
2. download our refined rectangle label and views from [GoogleDrive](https://drive.google.com/drive/folders/1vavvOjjd3nhs0fiTUpcR_As_dn3OvdCt?usp=sharing).
3. download the [pybullet_random](https://drive.google.com/drive/folders/1JRdD9OZ6nnZgcONfVFkJ7FiYn4m07Stz) .

    ```
    -- data
        -- planer_graspnet
            -- scenes
            -- depths
            -- rect_labels_filt_top10%_depth2_nms_0.02_10
            -- views
            -- models
            -- dex_models
        -- pybullet_random
            -- scenes
            -- rect_labels_filt_nms_0.02_10
    ```


## Training

For training GL-MSDA, the configuration files are in configs/sim_to_real/.

```shell script
python tools/train.py configs/graspnet/simb2realsense_source_only.py

CUDA_VISIBLE_DEVICES=0,1 .tools/dist_train.sh configs/graspnet/simb2realsense_source_only.py 2
```
## Testing

For testing  GL-MSDA, only support single-gpu inference.

```shell script
python tools/test_graspnet.py checkpoints/GL-MSDA/simb2realsense_fa.py checkpoints/GL-MSDA/simb2realsense_fa.pth --eval grasp
```

## Citation

If any part of our paper and repository is helpful to your work, please generously cite with:

```
@InProceedings{Ma_2024_ICRA,
    author    = {Haoxiang, Ma and Ran, Qin and Modi, Shi and Boyang, Gao and Huang, Di},
    title     = {Sim-to-Real Grasp Detection with Global-to-Local RGB-D Adaptation},
    booktitle = {International Conference on Robotics and Automation (ICRA)},
    year      = {2024}
```

