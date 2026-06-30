# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon

cv2.setNumThreads(0)

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='train', img_size=1024,
                 transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment = augment

        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = self.img_size
            self.query_featuremap_hw = (256, 256) 
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = self.img_size
            self.query_featuremap_hw = (256, 512)
        else:
            assert(False)

        self.rs_transform = albumentations.Compose([
            albumentations.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.4),
            albumentations.OneOf([
                albumentations.RandomRotate90(p=1),  
                albumentations.Rotate(limit=[180, 180], p=1),  
                albumentations.Rotate(limit=[270, 270], p=1),  
            ], p=0.75),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
        ], bbox_params=albumentations.BboxParams(format='pascal_voc'), additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[idx]
        
        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        bbox = bbox / (1024. / self.rs_wh)
        bbox = bbox.astype(int)
        
        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)

        mask_path = self.rsimg_dir.replace("/CVOGL/", "/CVOGL-Seg/")
        mask_name = "{}--bbox({},{},{},{}).jpg".format(rsimg_name[:-4], int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        mask_path = os.path.join(mask_path.replace("satellite", "mask-satellite"), mask_name)
        mask_rsimg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_rsimg = (mask_rsimg > 127).astype(np.float32)

        if self.augment:
            rs_transformed = self.rs_transform(image=rsimg, bboxes=[list(bbox)], mask=mask_rsimg)
            rsimg = rs_transformed['image']
            bbox = rs_transformed['bboxes'][0][0:4]
            mask_rsimg = rs_transformed['mask']

        # Lấy kích thước ảnh thực tế dạng numpy (1024x1024) trước khi chuyển qua Tensor
        h_img, w_img = queryimg.shape[0], queryimg.shape[1]
        
        # Tọa độ Click Point ban đầu trên hệ trục ảnh gốc 1024
        click_hw_high = (int(click_xy[1]), int(click_xy[0]))

        # Chạy Data Augmentation (Flip) trực tiếp trên ảnh OpenCV sạch sẽ
        flag = random.choice([True, False])
        if self.split_name == 'train' and flag:
            queryimg = cv2.flip(queryimg, 1) # Flip ảnh gốc nằm ngang dùng OpenCV
            click_hw_high = (click_hw_high[0], w_img - click_hw_high[1] - 1) # Đổi tọa độ click tương ứng

        # ---------------------------------------------------------------------
        # CẢI TIẾN CHÍNH: SINH HAI BẢN ĐỒ CLICK POINT ĐA NHIỆM (Độ phân giải 1024x1024)
        # ---------------------------------------------------------------------

        # 1. Nhánh Bounding Box: Tạo mặt nạ vùng bao vuông (Coarse Region Mask)
        click_box_mask = np.zeros((h_img, w_img), dtype=np.float32)
        box_radius = 60
        y_min = max(0, click_hw_high[0] - box_radius)
        y_max = min(h_img, click_hw_high[0] + box_radius)
        x_min = max(0, click_hw_high[1] - box_radius)
        x_max = min(w_img, click_hw_high[1] + box_radius)
        click_box_mask[y_min:y_max, x_min:x_max] = 1.0

        # 2. Nhánh Segmentation: Tạo bản đồ Gaussian Heatmap mịn (Fine-grained Center-focused)
        click_h_dist = [pow(one - click_hw_high[0], 2) for one in range(h_img)]
        click_w_dist = [pow(one - click_hw_high[1], 2) for one in range(w_img)]
        norm_factor = pow(h_img * h_img + w_img * w_img, 0.5)

        # Tạo lưới tọa độ tính nhanh thay vì dùng 2 vòng lặp lồng nhau làm chậm DataLoader
        dist_grid_h = np.array(click_h_dist, dtype=np.float32).reshape(-1, 1)
        dist_grid_w = np.array(click_w_dist, dtype=np.float32).reshape(1, -1)
        total_dist = np.sqrt(dist_grid_h + dist_grid_w)

        click_seg_heatmap = 1.0 - (total_dist / norm_factor)
        click_seg_heatmap = click_seg_heatmap * click_seg_heatmap # Tạo độ dốc sắc nét tại tâm
        click_seg_heatmap = click_seg_heatmap.astype(np.float32) # Đảm bảo kiểu float32 chuẩn PyTorch

        # Chuẩn hóa ảnh về dạng Tensor qua các hàm transform (torchvision) nếu có
        if self.transform is not None:
            rsimg = self.transform(rsimg.copy())
            queryimg = self.transform(queryimg.copy())

        return (queryimg,
                rsimg,
                torch.from_numpy(click_box_mask),
                torch.from_numpy(click_seg_heatmap),
                np.array(bbox, dtype=np.float32),
                mask_rsimg)