# Breaking Rectangular Shackles: Cross-View Object Segmentation for Fine-Grained Object Geo-Localization


# Dataset

The CVOGL-Seg dataset is created based on the CVOGL dataset; please download [CVOGL](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view?usp=sharing) first.

Our CVOGL-Seg dataset only provides the object segmentation mask; please download it from the [link](https://drive.google.com/file/d/1b3w0uY16uhkYI80d14pNMS2E4_ml7CAV/view?usp=sharing).

Download CVOGL and CVOGL-Seg and place them in the same directory, such as "/path/dataset/".

# Usage

## Train

Train on the **Drone → Satellite** task.

```shell
bash run_train_droneaerial.sh
```

```python
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --beta 1.0 --savename model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 24 --print_freq 50
```

Train on the **Ground → Satellite** task.

```shell
bash run_train_svi.sh
```

```python
python train.py --emb_size 768 --img_size 1024 --max_epoch 25 --data_root /path/dataset/CVOGL --data_name CVOGL_SVI --beta 1.0 --savename model_svi --gpu 0,1 --batch_size 12 --num_workers 8 --print_freq 50
```

## Evaluation

Evaluate on the **Drone → Satellite** task.

```shell
bash run_test_droneaerial.sh
```

```python
python train.py --val --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50
```

Evaluate on the **Ground → Satellite** task.

```shell
bash run_test_svi.sh
```

```python
python train.py --val --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_SVI --savename test_model_svi --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 768 --img_size 1024 --data_root /path/dataset/CVOGL --data_name CVOGL_SVI --savename test_model_svi --gpu 0,1 --batch_size 12 --num_workers 16 --print_freq 50
```

# SAM Prompt

Download the weight file from the link: [sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything)。

```python
python sam_prompt.py
```

# Model Zoo

| Task                 | Download Link                                                |
| -------------------- | ------------------------------------------------------------ |
| Drone *→* Satellite  | [link](https://drive.google.com/file/d/1YlWTVGiWNGEEb0b_4rU6RIqSQhLg5FX1/view?usp=sharing) |
| Ground *→* Satellite | [link](https://drive.google.com/file/d/1MDEpopjDWDpfbDd2Co0osCoWVAcEiGeN/view?usp=sharing) |




