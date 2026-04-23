import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img



def overlay_heatmap(img, heatmap, alpha=0.5):
    img = normalize(img)
    heatmap = normalize(heatmap)

    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))

    heatmap = plt.cm.jet(heatmap)[..., :3]
    overlay = img * (1 - alpha) + heatmap * alpha
    return overlay


def compute_error(pred, gt):
    # pred, gt: (B, 1, H, W)
    return ((pred - gt) ** 2).mean(dim=[1, 2, 3])


def get_worst_indices(pred, gt, topk=4):
    error = compute_error(pred, gt)
    _, idx = torch.topk(error, k=min(topk, len(error)))
    return idx

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().squeeze().numpy()
    elif isinstance(x, np.ndarray):
        x = x.squeeze()
    else:
        x = np.array(x)
    return x

def visualize_sample(q_img, r_img, pred, gt, save_path=None, idx=None):
    q_img = q_img.detach().cpu().permute(1, 2, 0).numpy()
    r_img = r_img.detach().cpu().permute(1, 2, 0).numpy()

    pred = to_numpy(pred)
    gt   = to_numpy(gt)

    pred = np.ascontiguousarray(pred)
    gt   = np.ascontiguousarray(gt)

    H, W = q_img.shape[:2]
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    gt   = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

    error_map = abs(pred - gt)

    overlay = overlay_heatmap(q_img, pred)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(normalize(q_img))
    axes[0].set_title("Query")

    axes[1].imshow(normalize(r_img))
    axes[1].set_title("Reference")

    axes[2].imshow(pred, cmap='jet')
    axes[2].set_title("Prediction")

    axes[3].imshow(gt, cmap='jet')
    axes[3].set_title("Ground Truth")

    axes[4].imshow(error_map, cmap='hot')
    axes[4].set_title("Error")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        name = f"sample_{idx}.png" if idx is not None else "sample.png"
        plt.savefig(os.path.join(save_path, name))
        plt.close()
    else:
        plt.show()


def visualize_worst_batch(query_imgs, reference_imgs, pred, gt, topk=4, save_path=None):
    """
    Visualize top-k worst predictions in a batch
    """

    idxs = get_worst_indices(pred, gt, topk)

    for i, idx in enumerate(idxs):
        visualize_sample(
            query_imgs[idx],
            reference_imgs[idx],
            pred[idx],
            gt[idx],
            save_path=save_path,
            idx=i
        )