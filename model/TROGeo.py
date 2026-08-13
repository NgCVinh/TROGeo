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


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models

from model.attention import SpatialTransformer

class SegToLocTokenFusion(nn.Module):
    def __init__(self, channels, num_tokens=8, num_heads=12):
        super(SegToLocTokenFusion, self).__init__()
        self.num_tokens = num_tokens
        
        # 1. Learnable Object Queries dùng để nén f_seg thành K Object Tokens
        self.object_queries = nn.Parameter(torch.randn(1, num_tokens, channels))
        
        # 2. Tokenization: Gom f_seg [B, HW, C] về K Object Tokens [B, K, C]
        self.mha_seg = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm_seg = nn.LayerNorm(channels)
        
        # 3. Thay MHA bằng SpatialTransformer chính chủ của bạn
        # f_bbox (2D) làm `x`, object_tokens làm `context`
        self.spatial_transformer_loc = SpatialTransformer(
            in_channels=channels, 
            n_heads=num_heads, 
            d_head=channels // num_heads, 
            depth=1, 
            context_dim=channels
        )
        
        # 4. Zero-Gating Residual Connection bảo vệ Baseline
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_seg, f_bbox):
        """
        f_seg: [B, C, H, W]
        f_bbox: [B, C, H, W]
        """
        B, C, H, W = f_bbox.shape
        
        # Step 1: Nén f_seg thành K Object Tokens [B, K, C]
        flat_seg = f_seg.flatten(2).transpose(1, 2)  # [B, H*W, C]
        q_obj = self.object_queries.expand(B, -1, -1)  # [B, K, C]
        
        obj_tokens, _ = self.mha_seg(query=q_obj, key=flat_seg, value=flat_seg)
        obj_tokens = self.norm_seg(obj_tokens)  # [B, K, C]
        
        # Step 2: f_bbox tương tác với K Object Tokens qua SpatialTransformer
        # x = f_bbox [B, C, H, W], context = obj_tokens [B, K, C]
        enhanced_bbox = self.spatial_transformer_loc(x=f_bbox, context=obj_tokens)
        
        # Step 3: Zero-Gated Residual Fusion
        f_fused_bbox = f_bbox + self.gamma * enhanced_bbox
        return f_fused_bbox

class TROGeo(nn.Module):
    def __init__(self, emb_size=768, num_object_tokens=8):
        super(TROGeo, self).__init__()

        base_model = SwinTBackbone()
        self.query_model = base_model
        self.reference_model = base_model
        self.combine_clickptns_conv = double_conv(4, 3)
        
        # Nhánh hỗ trợ thông tin Seg -> Loc TRƯỚC khi Cross-Attention với Reference (Cách 1)
        self.seg_to_loc_fusion = SegToLocTokenFusion(channels=emb_size, num_tokens=num_object_tokens, num_heads=12)

        # Spatial Transformers (Cross-Attention với Reference)
        self.cross_attention_seg = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size)
        self.cross_attention_loc = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1, context_dim=emb_size)

        # Head dự đoán Bounding Box (45 channels = 9 anchors * 5 params)
        self.fcn_out_box = nn.Sequential(
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
        
        q_feat = self.query_model(query_imgs)  # [B, C, H, W]
        r_feat = self.reference_model(reference_imgs)  # [B, C, H, W]

        # Khởi tạo đặc trưng phân nhánh từ Reference Backbone
        f_seg = r_feat
        f_bbox = r_feat

        # 2. Nhánh Seg hỗ trợ thông tin cho Bbox TRƯỚC khi Cross-Attention với Reference
        f_fused_bbox = self.seg_to_loc_fusion(f_seg=f_seg, f_bbox=f_bbox)

        # 3. Cross-Attention với Query Context
        context = rearrange(q_feat, 'b c h w -> b (h w) c').contiguous()
        
        seg_features = self.cross_attention_seg(x=f_seg, context=context)
        loc_features = self.cross_attention_loc(x=f_fused_bbox, context=context)

        # 4. Dự đoán đầu ra
        pred_box = self.fcn_out_box(loc_features)
        pred_mask = self.fcn_out_mask(seg_features)

        return pred_box, pred_mask