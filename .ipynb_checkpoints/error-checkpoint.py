import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_error_image(query_img, rs_img, gt_bbox, pred_bbox, pred_mask, iou_val, save_path):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    query_np = query_img.cpu().numpy() * std + mean
    query_np = np.clip(np.transpose(query_np, (1, 2, 0)), 0, 1)
    
    rs_np = rs_img.cpu().numpy() * std + mean
    rs_np = np.clip(np.transpose(rs_np, (1, 2, 0)), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Query Image
    axes[0].imshow(query_np)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # 2. RS Image (GT & Pred BBox)
    axes[1].imshow(rs_np)
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox[:4]
    p_x1, p_y1, p_x2, p_y2 = pred_bbox[:4]

    rect_gt = plt.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, 
                            fill=False, edgecolor='green', linewidth=2, label='GT')
    rect_pred = plt.Rectangle((p_x1, p_y1), p_x2 - p_x1, p_y2 - p_y1, 
                             fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Pred')
    axes[1].add_patch(rect_gt)
    axes[1].add_patch(rect_pred)
    axes[1].legend(loc='upper right')
    axes[1].set_title(f"RS Image (IoU: {iou_val:.3f})")
    axes[1].axis('off')

    # 3. Mask Pred
    mask_np = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    axes[2].imshow(mask_np, cmap='jet')
    axes[2].set_title("Pred Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def export_error_cases(query_imgs, rs_imgs, ori_gt_bbox, pred_bboxes, pred_coords, 
                       args, error_count, max_errors=100, iou_thresh=0.5):
    error_dir = os.path.join("./error_analysis", args.savename)
    os.makedirs(error_dir, exist_ok=True)

    for i in range(query_imgs.shape[0]):
        gt_box_i = ori_gt_bbox[i].cpu().numpy() if torch.is_tensor(ori_gt_bbox[i]) else ori_gt_bbox[i]
        pred_box_i = pred_bboxes[i].cpu().numpy() if torch.is_tensor(pred_bboxes[i]) else pred_bboxes[i]

        g_x1, g_y1, g_x2, g_y2 = gt_box_i[:4]
        p_x1, p_y1, p_x2, p_y2 = pred_box_i[:4]
        
        i_x1, i_y1 = max(g_x1, p_x1), max(g_y1, p_y1)
        i_x2, i_y2 = min(g_x2, p_x2), min(g_y2, p_y2)
        
        inter_area = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
        gt_area = max(0, g_x2 - g_x1) * max(0, g_y2 - g_y1)
        pred_area = max(0, p_x2 - p_x1) * max(0, p_y2 - p_y1)
        union_area = gt_area + pred_area - inter_area
        
        sample_iou = float(inter_area / union_area) if union_area > 0 else 0.0

        if sample_iou < iou_thresh and error_count < max_errors:
            error_count += 1
            save_path = os.path.join(error_dir, f"error_{error_count:04d}_iou_{sample_iou:.3f}.png")

            save_error_image(
                query_img=query_imgs[i],
                rs_img=rs_imgs[i],
                gt_bbox=gt_box_i,
                pred_bbox=pred_box_i,
                pred_mask=pred_coords[i],
                iou_val=sample_iou,
                save_path=save_path
            )

    return error_count