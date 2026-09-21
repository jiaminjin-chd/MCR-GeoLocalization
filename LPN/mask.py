import torch
class RandomPatchMask:
    def __init__(self, mask_ratio=0.75, patch_size=16, fill_value=0, keep_center=False, center_ratio=0.5):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.keep_center = keep_center
        self.center_ratio = center_ratio  # 中心区域占边长比例

    def __call__(self, img):
        C, H, W = img.shape
        p = self.patch_size
        device = img.device

        n_h, n_w = H // p, W // p
        num_patches = n_h * n_w

        if num_patches == 0:
            return img

        # 计算哪些 patch 属于中心区域
        if self.keep_center:
            center_h_start = int(n_h * (1 - self.center_ratio) / 2)
            center_h_end = int(n_h * (1 + self.center_ratio) / 2)
            center_w_start = int(n_w * (1 - self.center_ratio) / 2)
            center_w_end = int(n_w * (1 + self.center_ratio) / 2)

            # 创建中心区域掩码（这些 patch 必须保留）
            center_mask_2d = torch.zeros((n_h, n_w), device=device)
            center_mask_2d[center_h_start:center_h_end, center_w_start:center_w_end] = 1
            center_mask = center_mask_2d.flatten()  # (num_patches,)
        else:
            center_mask = torch.zeros(num_patches, device=device)

        # 计算要掩码的 patch 数量（仅在非中心区域进行）
        num_total_patches = num_patches
        num_center = center_mask.sum().item()
        num_non_center = num_total_patches - num_center
        num_mask_non_center = int(num_non_center * self.mask_ratio)

        # 生成所有非中心区域的索引
        all_indices = torch.arange(num_total_patches, device=device)
        non_center_indices = all_indices[center_mask == 0]

        # 从非中心区域随机选择要掩码的索引
        perm = torch.randperm(int(num_non_center), device=device)
        mask_non_center_indices = non_center_indices[perm[:num_mask_non_center]]

        # 构建最终掩码：1 表示保留，0 表示丢弃
        final_mask = torch.ones(num_total_patches, device=device)
        final_mask[mask_non_center_indices] = 0
        # 中心区域强制保留（其实 final_mask 已经为1，但为了安全可再赋值）
        final_mask[center_mask == 1] = 1

        # 重塑、上采样、应用掩码（同前）
        final_mask = final_mask.view(n_h, n_w)
        final_mask = final_mask.repeat_interleave(p, dim=0).repeat_interleave(p, dim=1)
        full_mask = torch.ones((H, W), device=device)
        full_mask[:n_h*p, :n_w*p] = final_mask

        out = img * full_mask + self.fill_value * (1 - full_mask)
        return out