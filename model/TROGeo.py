import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models

from model.attention import SpatialTransformer

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class SwinTBackbone(nn.Module):
    """Backbone trích xuất đặc trưng sử dụng Swin-Tiny (Swin-T)."""

    def __init__(self):
        super(SwinTBackbone, self).__init__()
        base_model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.features = base_model.features

    def forward(self, x):
        x = self.features(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TROGeo(nn.Module):
    def __init__(self, emb_size=768):
        super(TROGeo, self).__init__()

        # 1. Backbone & Input Prep
        base_model = SwinTBackbone()
        self.query_model = base_model
        self.reference_model = base_model
        self.combine_clickptns_conv = double_conv(4, 3)

        # 2. Spatial Transformers (Cross-Attention)
        self.cross_attention_seg = SpatialTransformer(
            in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size
        )
        self.cross_attention_loc = SpatialTransformer(
            in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size
        )

        # 3. Mask Refinement & Gate
        self.refine_mask = nn.Sequential(
            nn.Conv2d(emb_size, emb_size // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.gate_generator = nn.Sequential(
            nn.Conv2d(emb_size * 2, emb_size // 8, kernel_size=1),
            nn.BatchNorm2d(emb_size // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- HTC SPECIFIC MODULES (Khối tích hợp đặc trưng từ HTC) ---
        # Mask-to-Box Feature Integration Block: Chuyển đổi f_seg thành Feature cùng kích thước loc_features
        self.mask_to_box_flow = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(inplace=True)
        )

        # Head dự đoán Bounding Box Stage 1 (Coarse Box)
        self.fcn_out_box_stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 45, kernel_size=1),
        )

        # Head dự đoán Bounding Box Stage 2 (Refined Box theo HTC Style)
        self.fcn_out_box_stage2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 45, kernel_size=1),
        )

        # Head dự đoán Mask
        self.fcn_out_mask = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1)
        )

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        # 1. Trích xuất đặc trưng từ Backbone
        mat_clickptns = mat_clickptns.unsqueeze(1)
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        
        q_feat = self.query_model(query_imgs)        # [B, C, H, W]
        r_feat = self.reference_model(reference_imgs)  # [B, C, H, W]

        # Chuẩn bị Query context
        context = rearrange(q_feat, 'b c h w -> b (h w) c').contiguous()

        # 2. Nhánh Segmentation 
        f_seg = self.cross_attention_seg(x=r_feat, context=context) # [B, C, H, W]
        pred_mask = self.fcn_out_mask(f_seg)                          # [B, 1, H_mask, W_mask]

        # 3. Nhánh Localization cơ bản
        m_seg = self.refine_mask(f_seg) 
        gate_spatial = self.gate_generator(torch.cat([r_feat, f_seg], dim=1))
        r_feat_context_seg = r_feat * ((1.0 - gate_spatial) + gate_spatial * m_seg)
        
        loc_features_stage1 = self.cross_attention_loc(x=r_feat_context_seg, context=context) # [B, C, H, W]

        # Dự đoán Box Stage 1 (Dạng thô - dễ bị lỗi co cụm như ảnh của bạn)
        pred_box_stage1 = self.fcn_out_box_stage1(loc_features_stage1)

        # --- TƯ DUY HYBRID TASK CASCADE (HTC): MASK FEATURE FLOW ---
        # A. Trích xuất thông tin ngữ nghĩa & ranh giới từ f_seg
        mask_feat_flow = self.mask_to_box_flow(f_seg) # [B, C, H, W]

        # B. Dung hợp (Fusion) trực tiếp Mask Feature vào Loc Features (Dạng Residual Integration)
        loc_features_stage2 = loc_features_stage1 + mask_feat_flow

        # C. Dự đoán Box Stage 2 (Dự đoán lại dựa trên feature đã được Mask mở rộng)
        pred_box_stage2 = self.fcn_out_box_stage2(loc_features_stage2)
        # -------------------------------------------------------------

        if self.training:
            # Khi Train: Trả về cả 2 stages để tính Multi-stage Loss (Giúp hội tụ cực nhanh)
            return pred_box_stage1, pred_box_stage2, pred_mask
        else:
            # Khi Inference/Eval: Chỉ lấy kết quả đã tinh chỉnh tốt nhất ở Stage 2
            return pred_box_stage2, pred_mask