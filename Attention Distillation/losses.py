import math

import numpy as np
import torch
import torch.nn.functional as F

loss_fn = torch.nn.L1Loss()


def ad_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None, content_mask=None
):
    """
    content_mask: 用于mask loss计算的区域，只对mask区域计算loss
    格式: [B, 1, H, W]，1表示计算loss的区域，0表示不计算
    """
    loss = 0
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask=attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        
        # 如果提供了content_mask，只对mask区域计算loss
        if content_mask is not None:
            # 将self_out和target_out从[B, H*W, C] reshape为[B, C, H, W]以便应用mask
            # 处理不同的形状：可能是 [B, H*W, C] 或 [B, heads, H*W, C] 或其他形状
            shape = self_out.shape
            ndim = len(shape)
            
            try:
                if ndim == 3:
                    b, hw, c = shape[0], shape[1], shape[2]
                elif ndim == 4:
                    b, heads, hw, c = shape[0], shape[1], shape[2], shape[3]
                    # 如果是4维，reshape为3维：[B*heads, H*W, C]
                    self_out = self_out.view(b * heads, hw, c)
                    # 同样处理target_out
                    target_shape = target_out.shape
                    if len(target_shape) == 4:
                        target_out = target_out.view(b * heads, hw, c)
                    b = b * heads
                else:
                    # 如果形状不符合预期，跳过mask处理，使用普通loss
                    loss += loss_fn(self_out, target_out.detach())
                    continue
            except (ValueError, IndexError) as e:
                # 如果解包失败，使用普通loss
                loss += loss_fn(self_out, target_out.detach())
                continue
            
            h = w = int(np.sqrt(hw))
            
            # 调整mask尺寸，确保mask的batch维度匹配
            if content_mask.shape[0] == 1 and b > 1:
                # 如果mask只有一个batch，扩展到匹配
                content_mask_expanded = content_mask.repeat(b, 1, 1, 1)
            else:
                content_mask_expanded = content_mask
            
            mask_resized = F.interpolate(content_mask_expanded, size=(h, w), mode='bilinear', align_corners=False)
            mask_flat = mask_resized.view(b, 1, h * w)  # [B, 1, H*W]
            
            # 只对mask区域计算loss
            self_out_masked = self_out * mask_flat.transpose(1, 2)  # [B, H*W, C]
            target_out_masked = target_out.detach() * mask_flat.transpose(1, 2)  # [B, H*W, C]
            
            # 计算masked loss，并归一化到mask区域的数量
            mask_sum = mask_flat.sum() + 1e-8  # 避免除零
            # 如果mask区域太小，使用更稳定的归一化方式
            if mask_sum < (h * w) * 0.01:  # 如果mask区域小于1%，发出警告
                # 对于很小的mask区域，使用更保守的归一化
                loss += loss_fn(self_out_masked, target_out_masked) / (mask_sum / (h * w) + 1e-8)
            else:
                loss += (loss_fn(self_out_masked, target_out_masked) * (h * w) / mask_sum)
        else:
            loss += loss_fn(self_out, target_out.detach())
    return loss



def q_loss(q_list, qc_list, content_mask=None):
    """
    content_mask: 用于mask loss计算的区域，只对mask区域计算loss
    """
    loss = 0
    for q, qc in zip(q_list, qc_list):
        if content_mask is not None:
            # 将q和qc从[B, H*W, C] reshape为[B, C, H, W]以便应用mask
            # 处理不同的形状：可能是 [B, H*W, C] 或 [B, heads, H*W, C] 或其他形状
            shape = q.shape
            ndim = len(shape)
            
            try:
                if ndim == 3:
                    b, hw, c = shape[0], shape[1], shape[2]
                elif ndim == 4:
                    b, heads, hw, c = shape[0], shape[1], shape[2], shape[3]
                    # 如果是4维，reshape为3维：[B*heads, H*W, C]
                    q = q.view(b * heads, hw, c)
                    # 同样处理qc
                    qc_shape = qc.shape
                    if len(qc_shape) == 4:
                        qc = qc.view(b * heads, hw, c)
                    b = b * heads
                else:
                    # 如果形状不符合预期，跳过mask处理，使用普通loss
                    loss += loss_fn(q, qc.detach())
                    continue
            except (ValueError, IndexError) as e:
                # 如果解包失败，使用普通loss
                loss += loss_fn(q, qc.detach())
                continue
            
            h = w = int(np.sqrt(hw))
            
            # 调整mask尺寸，确保mask的batch维度匹配
            if content_mask.shape[0] == 1 and b > 1:
                # 如果mask只有一个batch，扩展到匹配
                content_mask_expanded = content_mask.repeat(b, 1, 1, 1)
            else:
                content_mask_expanded = content_mask
            
            mask_resized = F.interpolate(content_mask_expanded, size=(h, w), mode='bilinear', align_corners=False)
            mask_flat = mask_resized.view(b, 1, h * w)  # [B, 1, H*W]
            
            # 只对mask区域计算loss
            q_masked = q * mask_flat.transpose(1, 2)  # [B, H*W, C]
            qc_masked = qc.detach() * mask_flat.transpose(1, 2)  # [B, H*W, C]
            
            # 计算masked loss，并归一化到mask区域的数量
            mask_sum = mask_flat.sum() + 1e-8  # 避免除零
            # 如果mask区域太小，使用更稳定的归一化方式
            if mask_sum < (h * w) * 0.01:  # 如果mask区域小于1%，发出警告
                # 对于很小的mask区域，使用更保守的归一化
                loss += loss_fn(q_masked, qc_masked) / (mask_sum / (h * w) + 1e-8)
            else:
                loss += (loss_fn(q_masked, qc_masked) * (h * w) / mask_sum)
        else:
            loss += loss_fn(q, qc.detach())
    return loss

# weight = 200
def qk_loss(q_list, k_list, qc_list, kc_list):
    loss = 0
    for q, k, qc, kc in zip(q_list, k_list, qc_list, kc_list):
        scale_factor = 1 / math.sqrt(q.size(-1))
        self_map = torch.softmax(q @ k.transpose(-2, -1) * scale_factor, dim=-1)
        target_map = torch.softmax(qc @ kc.transpose(-2, -1) * scale_factor, dim=-1)
        loss += loss_fn(self_map, target_map.detach())
    return loss

# weight = 1
def qkv_loss(q_list, k_list, vc_list, c_out_list):
    loss = 0
    for q, k, vc, target_out in zip(q_list, k_list, vc_list, c_out_list):
        self_out = F.scaled_dot_product_attention(q, k, vc)
        loss += loss_fn(self_out, target_out.detach())
    return loss


def ad_loss_fragmented(
    q_list, ks_list, vs_list, self_out_list, 
    scale=1, source_mask=None, target_mask=None, content_mask=None,
    min_fragment_size=64, fragment_weight_decay=0.5
):
    """
    针对碎片化mask优化的attention distillation loss
    
    Args:
        min_fragment_size: 最小碎片大小（像素数），小于此值的碎片使用衰减权重
        fragment_weight_decay: 小碎片的权重衰减因子（0-1之间）
    
    原理:
        1. 对每个碎片分别计算loss
        2. 小碎片使用较小的权重，避免噪声影响
        3. 按碎片大小加权平均，确保大碎片主导优化方向
    """
    import numpy as np
    from scipy import ndimage
    
    loss = 0
    attn_mask = None
    
    # 检测碎片（只在第一层计算一次，假设所有层的mask形状相同）
    fragments_info = None
    if content_mask is not None:
        # 转换为numpy进行碎片检测
        mask_np = content_mask.squeeze().cpu().numpy() if content_mask.is_cuda or content_mask.requires_grad else content_mask.squeeze().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)
        labeled, num_features = ndimage.label(binary)
        
        # 计算每个碎片的大小
        fragments_info = []
        for i in range(1, num_features + 1):
            fragment_size = (labeled == i).sum()
            if fragment_size > 0:
                fragments_info.append({
                    'label': i,
                    'size': fragment_size,
                    'is_small': fragment_size < min_fragment_size
                })
    
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask = attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        
        if content_mask is not None:
            shape = self_out.shape
            ndim = len(shape)
            
            try:
                if ndim == 3:
                    b, hw, c = shape[0], shape[1], shape[2]
                elif ndim == 4:
                    b, heads, hw, c = shape[0], shape[1], shape[2], shape[3]
                    self_out = self_out.view(b * heads, hw, c)
                    target_shape = target_out.shape
                    if len(target_shape) == 4:
                        target_out = target_out.view(b * heads, hw, c)
                    b = b * heads
                else:
                    loss += loss_fn(self_out, target_out.detach())
                    continue
            except (ValueError, IndexError) as e:
                loss += loss_fn(self_out, target_out.detach())
                continue
            
            h = w = int(np.sqrt(hw))
            
            if content_mask.shape[0] == 1 and b > 1:
                content_mask_expanded = content_mask.repeat(b, 1, 1, 1)
            else:
                content_mask_expanded = content_mask
            
            mask_resized = F.interpolate(content_mask_expanded, size=(h, w), mode='bilinear', align_corners=False)
            mask_flat = mask_resized.view(b, 1, h * w)
            
            # 如果有碎片信息，使用碎片感知的loss计算
            if fragments_info and len(fragments_info) > 1:
                # 多碎片情况：对每个碎片分别计算
                fragment_losses = []
                fragment_weights = []
                
                # 将mask调整到当前分辨率
                mask_np_resized = mask_resized.squeeze().cpu().numpy() if mask_resized.is_cuda else mask_resized.squeeze().numpy()
                binary_resized = (mask_np_resized > 0.5).astype(np.uint8)
                labeled_resized, _ = ndimage.label(binary_resized)
                
                for fragment in fragments_info:
                    # 在当前分辨率下找到对应的碎片
                    fragment_mask_resized = (labeled_resized == fragment['label']).astype(float)
                    fragment_mask_flat = torch.from_numpy(fragment_mask_resized).float().view(1, 1, h * w)
                    fragment_mask_flat = fragment_mask_flat.to(self_out.device)
                    
                    if fragment_mask_flat.sum() > 0:
                        # 计算该碎片的loss
                        self_out_frag = self_out * fragment_mask_flat.transpose(1, 2)
                        target_out_frag = target_out.detach() * fragment_mask_flat.transpose(1, 2)
                        fragment_loss = loss_fn(self_out_frag, target_out_frag)
                        
                        # 计算权重：小碎片使用衰减权重
                        fragment_weight = fragment_mask_flat.sum().item()
                        if fragment['is_small']:
                            fragment_weight *= fragment_weight_decay
                        
                        fragment_losses.append(fragment_loss)
                        fragment_weights.append(fragment_weight)
                
                # 加权平均
                if fragment_weights:
                    total_weight = sum(fragment_weights) + 1e-8
                    weighted_loss = sum(loss * weight for loss, weight in zip(fragment_losses, fragment_weights)) / total_weight
                    # 归一化到总像素数
                    mask_sum = mask_flat.sum() + 1e-8
                    if mask_sum < (h * w) * 0.01:
                        loss += weighted_loss / (mask_sum / (h * w) + 1e-8)
                    else:
                        loss += weighted_loss * (h * w) / mask_sum
                else:
                    # 如果没有有效碎片，使用普通计算
                    self_out_masked = self_out * mask_flat.transpose(1, 2)
                    target_out_masked = target_out.detach() * mask_flat.transpose(1, 2)
                    mask_sum = mask_flat.sum() + 1e-8
                    if mask_sum < (h * w) * 0.01:
                        loss += loss_fn(self_out_masked, target_out_masked) / (mask_sum / (h * w) + 1e-8)
                    else:
                        loss += (loss_fn(self_out_masked, target_out_masked) * (h * w) / mask_sum)
            else:
                # 单碎片或碎片信息不可用，使用原始方法
                self_out_masked = self_out * mask_flat.transpose(1, 2)
                target_out_masked = target_out.detach() * mask_flat.transpose(1, 2)
                mask_sum = mask_flat.sum() + 1e-8
                if mask_sum < (h * w) * 0.01:
                    loss += loss_fn(self_out_masked, target_out_masked) / (mask_sum / (h * w) + 1e-8)
                else:
                    loss += (loss_fn(self_out_masked, target_out_masked) * (h * w) / mask_sum)
        else:
            loss += loss_fn(self_out, target_out.detach())
    
    return loss
