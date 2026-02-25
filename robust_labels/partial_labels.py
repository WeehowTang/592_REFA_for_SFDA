import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def partial_label_loss(sm_outputs, partial_Y, args, mask=None, smooth=-1, pos_coef=1.0):
    # refer to https://github.com/hongwei-wen/LW-loss-for-partial-label
    # LEVERAGED WEIGHTED LOSS FOR PARTIAL LABEL LEARNING
    # sm_outputs = nn.Softmax(dim=-1)(raw_outputs)
    onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1], device=args.device)
    onezero[partial_Y > 0] = 1  # selection of positive labels
    counter_onezero = 1 - onezero

    confidence = 1.0

    if smooth > 0:
        onezero = (1 - smooth) * onezero + smooth / args.class_num
    if mask is not None:
        selected_num = mask.sum()
        mask = mask.unsqueeze(1).expand_as(onezero).to(args.device)
    else:
        mask = torch.ones(sm_outputs.shape[0], sm_outputs.shape[1], device=args.device)
        selected_num = sm_outputs.shape[0]

    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = confidence * onezero * sig_loss1 * mask / onezero.sum(dim=-1, keepdim=True)
    average_loss1 = torch.sum(l1) / selected_num  # l1.size(0)
    average_loss = pos_coef * average_loss1

    return average_loss

# def partial_label_loss(
#         raw_outputs,
#         partial_Y,
#         args,
#         mask=None,             # mask[b] = 1 → uncertain ; 0 → certain
#         smooth=-0.0,
#         pos_coef=0.5,
# ):
#     """
#     Enhanced Partial-Label Loss with CERTAIN vs UNCERTAIN split
#     ------------------------------------------------------------------
#     - certain samples (mask=0) → standard CE using top-1 one-hot
#     - uncertain samples (mask=1) → partial-label (uniform top-K)
#     - combine both into a unified loss
#     """

#     sm_outputs = F.softmax(raw_outputs, dim=-1)    # [B, C]
#     B, C = sm_outputs.shape

#     # ------------------------------------------------------------------
#     # 1. Construct partial-label mask (multi-hot or one-hot)
#     #    partial_Y: [B, C] 0/1 mask
#     # ------------------------------------------------------------------
#     onezero = (partial_Y > 0).float()             # [B, C]

#     # Label smoothing
#     if smooth > 0:
#         onezero = (1 - smooth) * onezero + smooth / C

#     # ------------------------------------------------------------------
#     # 2. If mask is provided → select uncertain samples
#     # ------------------------------------------------------------------
#     if mask is not None:
#         mask = mask.float().to(raw_outputs.device)     # [B]
#         # uncertain = mask==1 ; certain = mask==0
#         mask_expanded = mask.unsqueeze(1).expand(B, C)
#         selected_uncertain = mask.sum().clamp(min=1)
#     else:
#         # fallback to all uncertain
#         mask = torch.ones(B, device=raw_outputs.device)
#         mask_expanded = mask.unsqueeze(1).expand(B, C)
#         selected_uncertain = B

#     # ------------------------------------------------------------------
#     # 3. uncertain samples → partial-label loss on onezero
#     # ------------------------------------------------------------------
#     # (A) compute per-class -log p
#     neg_log_p = -torch.log(sm_outputs + 1e-8)      # [B, C]

#     # (B) for uncertain samples, partial-label averaging
#     # per-sample positive loss: sum(-log p over partial labels) / k
#     pos_loss_per_sample = (onezero * neg_log_p).sum(dim=1) / (onezero.sum(dim=1) + 1e-8)   # [B]

#     # (C) apply uncertain mask
#     partial_loss = (pos_loss_per_sample * mask).sum() / selected_uncertain

#     # ------------------------------------------------------------------
#     # 4. certain samples → CE(one-hot)
#     # ------------------------------------------------------------------
#     # Determine top-1 class from partial label mask (the only 1 in onezero)
#     top1_labels = onezero.argmax(dim=1)   # [B]

#     ce_all = F.cross_entropy(raw_outputs, top1_labels, reduction="none")   # [B]

#     # apply certain mask (1-mask)
#     certain_mask = (1 - mask)
#     selected_certain = certain_mask.sum().clamp(min=1)

#     ce_loss = (ce_all * certain_mask).sum() / selected_certain

#     # ------------------------------------------------------------------
#     # 5. Combine (uncertain PL) + (certain CE)
#     # ------------------------------------------------------------------
#     total_loss = ce_loss + pos_coef * partial_loss

#     return total_loss

# Rolling (cumulative) selection of data points involved in the partial label loss
def selection_mask_bank_update(sample_selection_mask_bank, tar_idx, outputs, args, ratio=10):
    global logits_ratio
    if args.tau_type == 'fixed' or args.tau_type == 'stat':
        logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
        logits_ratio = logits[:, 0] / logits[:, 1]
        selected_indx = torch.where(logits_ratio < ratio)[0].to(tar_idx.device)
        selected_indx_bank = tar_idx[selected_indx]
        sample_selection_mask_bank[selected_indx_bank] = 1

    elif args.tau_type == 'cal':
        logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
        logits_max = logits[:, 0]
        logits_second = logits[:, 1]
        logits_ratio = logits[:, 0] / logits[:, 1]
        Num_classes = outputs.size()[1]
        eps2 = (1.0 / Num_classes)
        level_1 = 1.0 / Num_classes + eps2
        level_2 = logits_second + eps2
        selected_indx = torch.where(torch.logical_or(logits_max < level_1, logits_max < level_2))[0].to(tar_idx.device)
        selected_indx_bank = tar_idx[selected_indx]
        sample_selection_mask_bank[selected_indx_bank] = 1

    return sample_selection_mask_bank, logits_ratio[selected_indx]


def logits_ratio_calculation(outputs):
    logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
    logits_ratio = logits[:,0] / logits[:, 1]
    return logits_ratio.detach().cpu().numpy()


def obtain_sample_R_ratio(args, init_logits_ratio):
    if args.tau_type == "fixed":
        return args.sample_selection_R
    elif args.tau_type == "stat":
        return np.percentile(init_logits_ratio, [10])[0]
    else:
        return args.sample_selection_R

#####################################################################
# Auxiliary functions for autoUCon-SFDA (hyperparam self-generation)
#####################################################################

# Cumulative record of partial label set for each data point
def partial_label_bank_update(partial_label_bank, tar_idx, outputs, k_values):
    """
    partial_label_bank: [N, C] 的大 bank
    tar_idx: 当前 batch 对应 N 中的 index
    outputs: [bs, C] 概率（p_f 等）
    k_values: int 或 [bs] 每个样本的 k_i
    """

    device = outputs.device
    bs, C = outputs.size()

    # ---- Case 1: 固定 k（简单情况）----
    if isinstance(k_values, int):
        k = k_values
        # top-k indices
        _, topk_idx = torch.topk(outputs.detach(), k=k, dim=-1)

        # flatten row/col indices
        row_index = tar_idx.unsqueeze(1).expand_as(topk_idx).reshape(-1)
        col_index = topk_idx.reshape(-1)

        # 写入 bank（每行 top-k 填 1/k）
        partial_label_bank[row_index, col_index] = 1.0 / k
        return partial_label_bank


    # ---- Case 2: 动态 k_i ----
    # step 1: 找到每个样本的 sorted category indices
    sorted_idx = outputs.detach().argsort(dim=-1, descending=True)  # [bs, C]

    # step 2: 构建 mask，表示前 k_i 个类别为 True
    col_idx = torch.arange(C, device=device).unsqueeze(0).expand(bs, -1)
    mask = col_idx < k_values.view(-1, 1)   # [bs, C]

    # step 3: 取出 bank 中对应的行（tar_idx 展开到每个类别位置）
    expanded_rows = tar_idx.unsqueeze(1).expand_as(mask)  # [bs, C]

    # 选中的 bank 行（flatten）
    selected_rows = expanded_rows[mask]        # [sum(k_i)]
    # 选中的类别
    selected_cols = sorted_idx[mask]           # [sum(k_i)]

    # step 4: 每个选中位置的 1/k_i
    # mask 按 row 展开后，每组 k_i 次重复：需要取 tar_idx 对应的 k_i
    k_for_rows = k_values.repeat_interleave(k_values.max())  # ❌（不能用）

    # 正确做法：对 mask 展开后的每个元素取对应样本的 k_i
    k_selected = k_values.unsqueeze(1).expand_as(mask)[mask]  # [sum(k_i)]

    values = 1.0 / k_selected.float()

    # step 5: 写入 bank
    partial_label_bank[selected_rows, selected_cols] = values

    return partial_label_bank
# def partial_label_bank_update(partial_label_bank, tar_idx, outputs, k_values):
#     if isinstance(k_values, int):
#         _, topk_idx = torch.topk(outputs.detach(), k=k_values, dim=-1, largest=True)
        
#         # 展开行列索引
#         row_index = tar_idx.unsqueeze(1).expand_as(topk_idx).reshape(-1)
#         col_index = topk_idx.reshape(-1)

#         partial_label_bank[row_index, col_index] = 1 / k_values
#         return partial_label_bank

#     else:
#         batch_size, num_classes = outputs.size()
#         device = outputs.device

#         # 排序所有 probability
#         sorted_indices = outputs.detach().argsort(dim=-1, descending=True)  # [bs, C]

#         # 构建 mask：每行前 k_i 列为 True
#         col_indices = torch.arange(num_classes, device=device).unsqueeze(0)  # [1, C]
#         mask = (col_indices < k_values.view(-1, 1)).to(device)               # [bs, C]

#         # 将 tar_idx 映射到 partial_label_bank 的对应位置
#         tar_idx = tar_idx.to(device)
#         selected_rows = tar_idx.unsqueeze(1).expand_as(mask)[mask]  # [sum(k_i)]
#         selected_cols = sorted_indices[mask]                        # [sum(k_i)]

#         # 写入 bank（将对应的 top-k_i 位置置 1）
#         partial_label_bank[selected_rows, selected_cols] = 1 / k_values
        

#         return partial_label_bank



def calculate_k_values(outputs, k_max=2):
    batch_size, Num_classes = outputs.size()
    # eps1 = 2 / (Num_classes)
    eps1 = 0.1
    output_sorted, output_idx = torch.topk(outputs.detach().clone(), k=Num_classes, dim=1)  # bs * cls_num
    sorted_accumu_1 = torch.cat(
        [(torch.sum(output_sorted[:, :i + 1], dim=1) - eps1) / (i + 1) for i in range(Num_classes)]).reshape(
        Num_classes, batch_size).transpose(0, 1)  # [(128, ), ..., (128, )] -> (10, 128) -> (128, 10)
    max_values, k_star = torch.max(sorted_accumu_1, dim=1)
    k_star += 1  # skip 0
    k_values = torch.where(k_star < k_max, k_star, k_max)
    # print(k_star)
    # print(k_values)
    return k_star, k_values


def get_p_s_star_from_bank(partial_label_bank, idx, vlm_pred=None, label_aggregations=False, alpha=0.5):
    """
    partial_label_bank: [N, C] 0/1 mask for partial labels (p_s*)
    idx: index for batch samples
    vlm_pred: softmax probability of VLM (p_v)
    label_aggregations: whether to combine p_s* and p_v
    alpha: weight for p_s*
    """
    # vlm_pred = nn.Softmax(dim=-1)(vlm_pred)
    device = partial_label_bank.device

    # ---- Step1: get p_s* from bank (0/1 mask → uniform partial label) ----
    ps_mask = partial_label_bank[idx]        # shape [bs, C]

    # convert 0/1 mask → uniform partial label distribution
    ps_sum = ps_mask.sum(dim=1, keepdim=True).clamp(min=1)
    p_s_star = ps_mask.float() / ps_sum      # shape [bs, C]

    if not label_aggregations:
        return p_s_star.to(device)

    # ---- Step2: combine with VLM prediction p_v ----
    # p* = α p_s* + (1-α) p_v
    p_final = alpha * p_s_star + (1 - alpha) * vlm_pred

    # normalize (in case model is unstable)
    p_final = p_final / p_final.sum(dim=1, keepdim=True)

    return p_final.to(device)

def build_source_pseudo_label_banks(source_model, target_loader, args):
    source_model.eval()

    num_samples = len(target_loader.dataset)
    class_num = args.class_num

    partial_label_bank = torch.zeros(num_samples, class_num).long().to(args.device)
    pure_score_bank = torch.zeros(num_samples, class_num).float().to(args.device)
    sample_selection_mask_bank = torch.zeros(num_samples).long().to(args.device)

    with torch.no_grad():
        for images, _, idx in target_loader:

            images = images.to(args.device)
            idx = idx.to(args.device)   # [bs]

            # ------------------------------------------------
            # 1. Source model prediction
            # ------------------------------------------------
            logits_s = source_model(images).float()  # [bs, C]
            probs_s = nn.Softmax(dim=-1)(logits_s)

            # store soft probabilities
            pure_score_bank[idx] = probs_s   # [num_samples, C]

            # ------------------------------------------------
            # 2. Build partial labels p_s*
            # ------------------------------------------------
            _, k_values = calculate_k_values(probs_s, k_max=args.partial_k_max)
            partial_label_bank = partial_label_bank_update(
            partial_label_bank,
            idx,
            probs_s,
            k_values)
            # _, topk_idx = torch.topk(probs_s, k=args.partial_k_max, dim=1)  # [bs, k]

            # bs = idx.size(0)
            # k = args.partial_k_max

            # # 展开行和列索引
            # row_index = idx.unsqueeze(1).expand(-1, k).reshape(-1)   # [bs*k]
            # col_index = topk_idx.reshape(-1)                         # [bs*k]

            # partial_label_bank[row_index, col_index] = 1  # 赋值 1

            # ------------------------------------------------
            # 3. Compute uncertain mask
            # ------------------------------------------------
            top2 = torch.topk(probs_s, k=2, dim=1).values
            ratio = top2[:, 0] / (top2[:, 1] + 1e-8)

            uncertain_mask = ratio < args.sample_selection_R
            sample_selection_mask_bank[idx] = uncertain_mask.long()

    return pure_score_bank, partial_label_bank, sample_selection_mask_bank



