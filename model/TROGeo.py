import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models

from model.attention import SpatialTransformer, FeatureGating, IterativeRefinementHead

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class SwinSBackbone(nn.Module):
    def __init__(self):
        super(SwinSBackbone, self).__init__()
        # Trích xuất đặc trưng tiêu chuẩn từ mô hình pre-train
        base_model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        self.features = base_model.features

    def forward(self, x):
        # Đầu ra gốc: [B, H/32, W/32, 768] -> Chuyển về chuẩn: [B, 768, H/32, W/32]
        x = self.features(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TROGeo(nn.Module):
    def __init__(self, emb_size=768):
        super(TROGeo, self).__init__()

        # 1. Khởi tạo Backbone Swin-S dùng chung để tiết kiệm bộ nhớ
        self.backbone = SwinSBackbone()
        
        # 2. TÁCH BIỆT: Khối tích hợp 4 kênh (RGB + Click Point) độc lập cho 2 nhánh nhiệm vụ
        self.combine_box_click = double_conv(4, 3)  # Dành cho nhánh học cấu trúc Bounding Box
        self.combine_seg_click = double_conv(4, 3)  # Dành cho nhánh học ranh giới mịn (Segmentation)

        # 3. Các module Attention và Gating
        self.gating = FeatureGating(in_channels=emb_size)
        self.cross_attention = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size)

        # 4. Mạch giải nén không gian tuần tự (Từ 1/32 lên 1/16 rồi lên 1/8) cho đầu Refinement
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(emb_size, emb_size // 2, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(emb_size // 2, emb_size // 4, kernel_size=4, stride=2, padding=1) 
        )
        self.iterative_coords = IterativeRefinementHead(in_channels=emb_size // 4, num_steps=2)
        
        # 5. Head dự đoán Bounding Box nguyên bản từ đặc trưng thô
        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 45, kernel_size=1),
        )
        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )

    def forward(self, query_imgs, reference_imgs, click_box_mask, click_seg_heatmap):
        """
        Args:
            query_imgs: Ảnh Drone [B, 3, 1024, 1024]
            reference_imgs: Ảnh Vệ tinh [B, 3, 1024, 1024]
            click_box_mask: Tensor nhị phân vùng bao của điểm click [B, 1024, 1024]
            click_seg_heatmap: Tensor Gaussian loang mờ của điểm click [B, 1024, 1024]
        """
        # --- BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG ĐƠN LẺ CHO ẢNH VỆ TINH ---
        r_feat = self.backbone(reference_imgs) # Kích thước chuẩn: [B, 768, 32, 32]

        # Chuẩn bị chiều không gian cho các bản đồ Click Point [B, 1024, 1024] -> [B, 1, 1024, 1024]
        click_box_mask = click_box_mask.unsqueeze(1)
        click_seg_heatmap = click_seg_heatmap.unsqueeze(1)

        # --- BƯỚC 2: NHÁNH DỰ ĐOÁN BOUNDING BOX (BOX BRANCH) ---
        # Hòa trộn ảnh Drone với mặt nạ vùng bao (Box Mask)
        q_box_input = self.combine_box_click(torch.cat((query_imgs, click_box_mask), dim=1))
        q_box_feat = self.backbone(q_box_input)
        
        # Ép đặc trưng Box của Drone với ảnh Vệ tinh qua Gating & Cross-Attention
        #r_feat_gated_box = self.gating(q_box_feat, r_feat)
        context_box = rearrange(q_box_feat, 'b c h w -> b (h w) c').contiguous()

        #fused_box = self.cross_attention(x=r_feat_gated_box, context=context_box)
        fused_box = self.cross_attention(x=r_feat, context=context_box)
        # Xuất kết quả dự đoán Box
        outbox = self.fcn_out(fused_box)

        # --- BƯỚC 3: NHÁNH TINH CHỈNH TỌA ĐỘ / PHÂN VÙNG (REFINEMENT BRANCH) ---
        # Hòa trộn ảnh Drone với bản đồ Gaussian mịn (Segmentation Heatmap)
        q_seg_input = self.combine_seg_click(torch.cat((query_imgs, click_seg_heatmap), dim=1))
        q_seg_feat = self.backbone(q_seg_input)
        
        # Ép đặc trưng Phân vùng của Drone với ảnh Vệ tinh qua Gating & Cross-Attention
        #r_feat_gated_seg = self.gating(q_seg_feat, r_feat)
        context_seg = rearrange(q_seg_feat, 'b c h w -> b (h w) c').contiguous()
        fused_seg = self.cross_attention(x=r_feat, context=context_seg)
        
        # Phóng đại độ phân giải đặc trưng tuần tự và đưa qua Iterative Refinement nắn tọa độ
        #fused_high_res = self.upsample(fused_seg)
        #coords_list = self.iterative_coords(fused_high_res)
        coodrs = self.coodrs_out(fused_seg)
        return outbox, coodrs