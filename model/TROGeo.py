# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.attention import SpatialTransformer, FeatureGating, IterativeRefinementHead

import numpy as np
import torchvision.models as models

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        self.model.avgpool = None
        self.model.head = None
        self.model.norm = None

    def forward(self, x):
        x = self.model.features(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class TROGeo(nn.Module):
    def __init__(self, emb_size=768):
        super(TROGeo, self).__init__()

        # ... các phần Backbone và Gating giữ nguyên ...
        base_model = SwinTransformer()
        self.query_model = base_model
        self.reference_model = base_model
        self.combine_clickptns_conv = double_conv(4, 3)
        self.gating = FeatureGating(in_channels=emb_size)
        self.cross_attention = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size)

        # Upsample đặc trưng lên độ phân giải cao hơn để dự đoán chính xác hơn
        self.upsample = nn.ConvTranspose2d(emb_size, emb_size // 4, kernel_size=4, stride=2, padding=1)

        # THAY THẾ: Cơ chế Refinement 4 bước
        self.iterative_coords = IterativeRefinementHead(in_channels=emb_size // 4, num_steps=1
        )
        
        # Head dự đoán box (nếu bạn vẫn cần)
        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 45, kernel_size=1),
        )

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        # 1. Feature Extraction & Gating
        mat_clickptns = mat_clickptns.unsqueeze(1)
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        
        q_feat = self.query_model(query_imgs)
        r_feat = self.reference_model(reference_imgs)
        
        # Gating lọc thông tin đa quy mô
        r_feat_gated = self.gating(q_feat, r_feat)

        # 2. Cross-Attention
        context = rearrange(q_feat, 'b c h w -> b (h w) c').contiguous()
        fused_features = self.cross_attention(x=r_feat_gated, context=context)

        # 3. Iterative Refinement
        # Bước 1: Cho đặc trưng qua lớp upsample để lấy lại độ chi tiết không gian
        fused_high_res = self.upsample(fused_features)
        
        # Bước 2: Suy luận qua 3 Heads
        # coords_list sẽ chứa [Head_1_Coarse, Head_2_Geometric, Head_3_Fine]
        coords_list = self.iterative_coords(fused_high_res)

        # Kết quả box vẫn có thể dùng fused_features gốc
        outbox = self.fcn_out(fused_features)

        return outbox, coords_list # Trả về list kết quả để tính Multi-step Loss