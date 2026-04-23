# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.attention import SpatialTransformer, FeatureGating, LearnablePolarTransform

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

class FeatureUncertaintyRefinement(nn.Module):
    """Module tinh chỉnh đặc trưng qua từng vòng lặp dựa trên độ bất định"""
    def __init__(self, dim):
        super().__init__()
        # Mạng sinh ra delta_feature và uncertainty_map
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1) 
        )

    def forward(self, feat):
        out = self.refine_conv(feat)
        delta_feat, uncertainty = out.chunk(2, dim=1) # Tách làm 2 nửa

        # Trọng số tinh chỉnh: Vùng uncertainty cao -> weight tiến về 1 (tinh chỉnh mạnh)
        # Vùng uncertainty thấp (đã tự tin) -> weight tiến về 0 (giữ nguyên)
        weight = torch.sigmoid(uncertainty) 

        # Cập nhật đặc trưng
        refined_feat = feat + (delta_feat * weight)
        return refined_feat

class TROGeo(nn.Module):
    def __init__(self, emb_size=768, num_refine_steps=3):
        super(TROGeo, self).__init__()

        base_model = SwinTransformer()
        self.query_model = base_model
        self.reference_model = base_model
        self.num_refine_steps = num_refine_steps
        self.combine_clickptns_conv = double_conv(4, 3)
        # THÊM: Khởi tạo module Gating
        #self.gating = FeatureGating(in_channels=emb_size)

        #self.polar_transform = LearnablePolarTransform(in_channels=emb_size)

        self.cross_attention = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1,
                                                  context_dim=emb_size)

        # THÊM: Vòng lặp tinh chỉnh (Iterative Refinement Loop)
        self.refinement_steps = nn.ModuleList([
            FeatureUncertaintyRefinement(dim=emb_size) for _ in range(num_refine_steps)
        ])

        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_size // 2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, emb_size // 4, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 4, 9 * 9, kernel_size=1),
        )
        
        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )


    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)

        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))

        query_fvisu = self.query_model(query_imgs)
        reference_fvisu = self.reference_model(reference_imgs)

        #Áp dụng Gating: Dùng thông tin từ Query (có chứa click point) 
        #để làm nổi bật/lọc các vùng tương ứng trên Reference

        #reference_fvisu_gated = self.gating(query_fvisu, reference_fvisu)

        #query_polar = self.polar_transform(query_fvisu) # Chuyển cả Query để đồng bộ không gian

        context = rearrange(query_fvisu, 'b c h w -> b (h w) c').contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)

        current_feat = fused_features
        for i in range(self.num_refine_steps):
            current_feat = self.refinement_steps[i](current_feat)

        outbox = self.fcn_out(current_feat)
        coodrs = self.coodrs_out(current_feat)

        return outbox, coodrs
