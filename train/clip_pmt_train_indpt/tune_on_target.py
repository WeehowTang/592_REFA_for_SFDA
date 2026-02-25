import os
import torch
import torch.nn as nn
import torch.optim as optim
from LoadDataset import *
from Custom_PL import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import re


def _get_hf_classnames(train_ds, fallback_num_classes=126):
    """
    Try to get class names from HuggingFace Dataset features.
    """
    try:
        feat = train_ds.features.get("label", None)
        if feat is not None and hasattr(feat, "names") and feat.names is not None:
            if len(feat.names) > 0:
                return list(feat.names)
    except Exception:
        pass
    # fallback
    return [str(i) for i in range(fallback_num_classes)]

def compute_class_balance_loss(logits):
    probs = F.softmax(logits, dim=1)
    cls_avg = probs.mean(dim=0)
    balance_loss = -torch.sum(cls_avg * torch.log(cls_avg + 1e-8))
    return balance_loss

def mutual_information_loss(logits):
    """
    Compute MI loss = H(Y|X) - H(Y)
    logits: (B, C)
    """
    probs = F.softmax(logits, dim=1)           
    # conditional entropy: average prediction uncertainty per sample
    conditional_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
    # marginal entropy: entropy of mean prediction across batch
    marginal_probs = probs.mean(dim=0)
    marginal_entropy = -torch.sum(marginal_probs * torch.log(marginal_probs + 1e-8))

    mi_loss = marginal_entropy-conditional_entropy
    return -mi_loss

def generate_pseudo_labels(logits, threshold=0.7):
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)

    mask = max_probs.ge(threshold).float()
    return pseudo_labels, mask


def tuning_prompt_model_on_target(
        model: ClipTuningModel,
        train_dataset,
        vali_view: bool = False,
        dtype=torch.float32,
        class_map=None,
        target_domain: str = '',
        batch_size: int = 32,
        lr: float = 1e-3,
        n_epochs: int = 50,
        save_epc: int = 5,
        device: str = "cuda",
        scheduler_type: str = None,
        save_dir: str = "./checkpoints",
        output_img: str = "",
        start_epoch: int = 0,
        optimizer_state: str = None,
        balance_ratio: float = 0.5,  # control class balance strength
        collate_fn = None,
):
    # --- 构建 class2idx ---
    if isinstance(class_map, dict):
        class2idx = class_map
        idx2class = {v: k for k, v in class_map.items()}
    elif isinstance(class_map, list):
        class2idx = {cls: i for i, cls in enumerate(class_map)}
        idx2class = {i: cls for i, cls in enumerate(class_map)}
    else:
        raise ValueError("class_map must be list or dict")

    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vali_loader = train_loader if vali_view else None

    # optimizer = torch.optim.AdamW(model.prompt_learner.parameters(),lr=lr)
    optimizer = optim.SGD(model.prompt_learner.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        # scheduler
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        scheduler = None

    model.to(device, dtype=dtype)
    model.train()
    os.makedirs(output_img, exist_ok=True)
    # for batch_idx, batch in enumerate(train_loader):
        
    #     # ---- unify batch format (HF dict / old tuple) ----
    #     if isinstance(batch, dict):
    #         images = batch.get("pixel_values", batch.get("images"))
    #         labels = batch.get("label", batch.get("labels"))
    #         idx = batch.get("idx", None)
    #     else:
    #         images, labels, idx = batch
    
    #     images = images.to(device, dtype=dtype)

    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        class_correct = {cls: 0 for cls in class2idx.keys()}
        class_total = {cls: 0 for cls in class2idx.keys()}
        for batch_idx, batch in enumerate(train_loader):
            # HF dict batch
            if isinstance(batch, dict):
                images = batch["pixel_values"]
                labels = batch.get("label", batch.get("labels"))
                idx = batch.get("idx", None)
            else:
                # old tuple batch
                images, labels, idx = batch
        
            images = images.to(device, dtype=dtype)
            if isinstance(labels, (list, tuple)) and len(labels) > 0 and isinstance(labels[0], str):
                y_true = torch.tensor([class2idx[cls] for cls in labels], dtype=torch.long, device=device)
            else:
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.long)
                y_true = labels.to(device, dtype=torch.long)
                
        # for batch_idx, (images, labels, _) in enumerate(train_loader):
        #     images = images.to(device, dtype=dtype)
        #     if isinstance(labels, (list, tuple)) and len(labels) > 0 and isinstance(labels[0], str):
        #         y_true = torch.tensor([class2idx[cls] for cls in labels], dtype=torch.long, device=device)
        #     else:
        #         # HF returns torch.Tensor already, or list[int]
        #         y_true = torch.as_tensor(labels, dtype=torch.long, device=device)
                
            # if isinstance(labels[0], str):
            #     y_true = torch.tensor([class2idx[cls] for cls in labels], dtype=torch.long, device=device)
            # else:
            #     y_true = labels.to(device, dtype=torch.long)
            # loss
            optimizer.zero_grad()

            logits = model(images).float()
            
            # logits = F.softmax(logits / 0.5, dim=-1)
            # pseudo_labels, mask = generate_pseudo_labels(logits, threshold=tau)
            # loss_ce = criterion(logits, pseudo_labels)
            # loss_cm = (loss_ce * mask).sum() / (mask.sum() + 1e-8)
            loss = mutual_information_loss(logits)
            # loss = loss_ce + balance_ratio * balance_loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
            # per-class acc
            for label, pred in zip(y_true, preds):
                cls_name = list(class2idx.keys())[list(class2idx.values()).index(label.item())]
                class_total[cls_name] += 1
                if label.item() == pred.item():
                    class_correct[cls_name] += 1
            # accuracy
            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch + 1}/{n_epochs}] Step {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        if scheduler is not None:
            scheduler.step()
        acc = correct / total
        print(f"Tuned Accuracy: {acc * 100:.4f}%")

        
        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}/{n_epochs}] Average Loss: {avg_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        class_acc = {
                cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0)
                for cls in class2idx.keys()
            }
        print("📊 Per-class Accuracy:")
        for cls, a in class_acc.items():
            print(f" {cls:<15}: {a*100:.2f} %  ({class_correct[cls]}/{class_total[cls]})")
        # plot results
        plt.figure(figsize=(10, 5))
        plt.bar(class_acc.keys(), class_acc.values(), color="skyblue")
        plt.axhline(y=acc, color="red", linestyle="--", label=f"Overall Acc={acc:.4f}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title(f"Per-class Accuracy of clip model on {target_domain} domain")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_img, f'clip model acc_{acc}.jpg'))
        plt.close()
    
        if (epoch + 1) % save_epc == 0 and save_dir is not None:
            ckpt_path = os.path.join(save_dir, f"Learnable_parameters_epoch{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "batch_size": model.bs,
                "model_state": model.prompt_learner.state_dict(),  # 只保存 PromptLearner
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

def tune_on_target_main(args):
    os.makedirs(args["save_dir"], exist_ok=True)

    domain = args.get("target domain", "target")
    print(f"Starting CLIP Prompt Training on {domain}...")

    device = args["device"]
    dtype = args["dtype"]
    save_dirs = args["save_dir"]
    bs, epochs, lr = args["batchsize"], args["epochs"], args["lr"]
    lr_scheduler, save_epc = args["lr_type"], args["n_epc_save"]
    archive_clip, ctx_init, n_ctx = args["clip_arch"], args["ctx_init"], args["n_ctx"]
    ckpt_paths = args.get("ckpt_dir", None)
    img_save_dir = args.get("output_img_dir", None)

    # -------------------------
    # 1) Build dataset interface
    # -------------------------
    use_hf = ("train_ds" in args) and (args["train_ds"] is not None)
    collate_fn = args.get("collate_fn", None)
    if use_hf:
        # HF dataset path: already loaded via load_dataset(...) in your main()
        train_dataset = args["train_ds"]
        val_dataset = args.get("val_ds", None)

        # (Optional) subset
        indices = args["subset"] if args.get("subset", None) else None
        if indices is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            if val_dataset is not None:
                val_dataset = torch.utils.data.Subset(val_dataset, indices)

        # classnames from HF features (DomainNet-126 -> should be 126)
        classnames = args.get("classnames", None)
        if classnames is None:
            # try infer from hf features; fallback to 126
            classnames = _get_hf_classnames(args["train_ds"], fallback_num_classes=args.get("num_classes", 126))
        class2idx = {c: i for i, c in enumerate(classnames)}

        print(f"[HF] Train size: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"[HF] Val size: {len(val_dataset)}")
        else:
            print("[HF] Val dataset: None (will use train as val in your downstream if needed)")

    else:
        # Original local-path dataset logic (Office31Dataset etc.)
        # --- Load DSLR test dataset ---
        init_dataset = Office31Dataset(args["target_img_dir"], size=args["size"])
        classnames = init_dataset.classes

        # We'll create target_dataset below after model is built (needs clip_preprocess)
        train_dataset = None
        val_dataset = None
        class2idx = None

    # -------------------------
    # 2) Resume checkpoint
    # -------------------------
    start_epoch = 0
    optimizer_state = None
    latest_ckpt = None

    if args.get("resume", False):
        if ckpt_paths is None or (not os.path.exists(ckpt_paths)):
            print("[WARN] resume=True but ckpt_dir not found. Training from scratch.")
        else:
            ckpts = [f for f in os.listdir(ckpt_paths) if f.endswith(".pt")]
            if ckpts:
                ckpts = sorted(ckpts, key=lambda x: int(x.split("epoch")[-1].split(".pt")[0]))
                latest_ckpt = os.path.join(ckpt_paths, ckpts[-1])
                latest_epoch = int(ckpts[-1].split("epoch")[-1].split(".pt")[0])
                print("Latest checkpoint:", latest_ckpt, "epoch:", latest_epoch)

                ckpt = torch.load(latest_ckpt, map_location=device)
                optimizer_state = ckpt.get("optimizer_state", None)
                bs_vlm = ckpt.get("batch_size", bs)
                start_epoch = latest_epoch + 1

                # Load model with checkpoint
                model, clip_preprocess = get_load_clip_tuning_model(
                    clip_arch=archive_clip,
                    classnames=classnames,
                    device=device,
                    load=True,
                    n_ctx=n_ctx,
                    ctx_position=args["ctx_pos"],
                    ctx_init=ctx_init,
                    batch_size=bs_vlm,
                    dtype=dtype,
                    learned_cls=False,
                    ckpt_path=latest_ckpt
                )
                print(f"Loaded checkpoint. batch_size={bs}, resume from epoch {start_epoch}")
            else:
                print("[WARN] No checkpoint found in ckpt_dir. Training from scratch.")

    # -------------------------
    # 3) Build model if not resumed
    # -------------------------
    if (not args.get("resume", False)) or (latest_ckpt is None):
        print(f"🚀 Training from scratch. Batchsize={bs}, lr={lr}, start={start_epoch}, epochs={epochs}")
        model, clip_preprocess = get_load_clip_tuning_model(
            clip_arch=archive_clip,
            classnames=classnames,
            device=device,
            load=False,
            n_ctx=n_ctx,
            ctx_position=args["ctx_pos"],
            ctx_init=ctx_init,
            batch_size=None,
            dtype=dtype,
            learned_cls=False
        )

    # -------------------------
    # 4) If local-path dataset: now create dataset with clip_preprocess
    # -------------------------
    if not use_hf:
        target_dataset = Office31Dataset(args["target_img_dir"], preprocess=clip_preprocess)
        indices = args["subset"] if args.get("subset", None) else None
        if indices is not None:
            target_dataset = torch.utils.data.Subset(target_dataset, indices)

        train_dataset = target_dataset
        val_dataset = None  # you used train as val previously
        class2idx = {cls: i for i, cls in enumerate(target_dataset.classes)}

        print(f"[Local] Train size: {len(train_dataset)} (Validation uses train in your old code)")

    # -------------------------
    # 5) Final sanity checks (very helpful for DN126)
    # -------------------------
    try:
        sample = train_dataset[0]
        # expected keys: pixel_values/label/idx
        if isinstance(sample, dict) and "label" in sample:
            lab = sample["label"]
            # show a quick range check if it's a torch tensor
            print("[Sanity] sample keys:", list(sample.keys()))
            print("[Sanity] sample label:", lab)
    except Exception as e:
        print("[WARN] Sanity check failed:", e)

    # -------------------------
    # 6) Train
    # -------------------------
    tuning_prompt_model_on_target(
        model=model,
        train_dataset=train_dataset,
        class_map=class2idx,
        dtype=dtype,
        n_epochs=epochs,
        batch_size=bs,
        lr=lr,
        vali_view=True,
        target_domain=domain,
        output_img=img_save_dir,
        scheduler_type=lr_scheduler,
        save_dir=save_dirs,
        save_epc=save_epc,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        collate_fn=collate_fn,
    )

# def tune_on_target_main(args):
#     os.makedirs(args["save_dir"], exist_ok=True)
#     domain = args['target domain']
#     print(f"Starting CLIP Prompt Training on {domain}...")
#     device = args['device']
#     dtype = args['dtype']
#     save_dirs, bs, epochs, lr = args['save_dir'], args['batchsize'], args['epochs'], args['lr']
#     lr_scheduler, save_epc = args['lr_type'], args['n_epc_save']
#     archive_clip, ctx_init, n_ctx = args['clip_arch'], args['ctx_init'], args['n_ctx']
#     cond_prob, cls_blc = args['conf prob'], args['balance ratio']
#     ckpt_paths, img_save_dir = args['ckpt_dir'], args['output_img_dir']
#     # --- Load DSLR test dataset ---
#     init_dataset = Office31Dataset(args['target_img_dir'], size=args['size'])

#     # --- resume checkpoint ---
#     start_epoch = 0
#     optimizer_state = None
#     if args['resume']:
#         ckpts = [f for f in os.listdir(ckpt_paths) if f.endswith(".pt")]
#         if ckpts:
#             # 按 epoch 排序
#             ckpts = sorted(ckpts, key=lambda x: int(x.split("epoch")[-1].split(".pt")[0]))
#             latest_ckpt = os.path.join(ckpt_paths, ckpts[-1])
#             latest_epoch = int(ckpts[-1].split("epoch")[-1].split(".pt")[0])
#             print("Latest checkpoint:", latest_ckpt, "epoch:", latest_epoch)

#             ckpt = torch.load(latest_ckpt, map_location=device)
#             optimizer_state = ckpt.get("optimizer_state", None)
#             bs = ckpt.get("batch_size", bs)
#             # start_epoch = ckpt.get("epoch", 0)
#             start_epoch = latest_epoch + 1
#             # --- Load Tuning Model ---
#             model, clip_preprocess = get_load_clip_tuning_model(
#                 clip_arch=archive_clip,
#                 classnames=init_dataset.classes,
#                 device=device,
#                 load=True,
#                 n_ctx=n_ctx,
#                 ctx_position=args['ctx_pos'],
#                 ctx_init=ctx_init,
#                 batch_size=None,
#                 dtype=dtype,
#                 learned_cls=False,
#                 ckpt_path=latest_ckpt
#             )
#             # # 加载 ctx 和 cls
#             # if "ctx" in ckpt:
#             #     model.prompt_learner.ctx.data = ckpt["ctx"].to(device)
#             # if ckpt.get("cls") is not None and hasattr(model.prompt_learner,"cls"):
#             #     model.prompt_learner.cls.data = ckpt["cls"].to(device)

#             # 加载 optimizer 状态

#             print(f"Loaded ctx, cls, batch_size={bs}, resume from epoch {start_epoch}")
#         else:
#             print("No checkpoint found, training from scratch.")

#     else:
#         print(f"🚀 Training from scratch.Batchsize is {bs}, lr is {lr}, start from {start_epoch} and end for {epochs}")
#         model, clip_preprocess = get_load_clip_tuning_model(
#             clip_arch=archive_clip,
#             classnames=init_dataset.classes,
#             device=device,
#             load=False,
#             n_ctx=n_ctx,
#             ctx_position=args['ctx_pos'],
#             ctx_init=ctx_init,
#             batch_size=None,
#             dtype=dtype,
#             learned_cls=False)

#     target_dataset = Office31Dataset(args["target_img_dir"], preprocess=clip_preprocess)
#     class2idx = {cls: i for i, cls in enumerate(target_dataset.classes)}
#     # vali_dataset = Office31Dataset(args['vali_img_dir'], preprocess=clip_preprocess)
#     indices = args['subset'] if args['subset'] else None
#     target_dataset = torch.utils.data.Subset(target_dataset, indices) if indices else target_dataset
#     print(f"Train size: {len(target_dataset)}, Validation size: {len(target_dataset)}")

#     # vali_dataset = torch.utils.data.Subset(vali_dataset, indices) if indices else vali_dataset
#     tuning_prompt_model_on_target(
#         model=model,
#         train_dataset=target_dataset,
#         class_map=class2idx,
#         dtype=dtype,
#         n_epochs=epochs,
#         batch_size=bs,
#         lr=lr,
#         vali_view=True,
#         target_domain=domain,
#         output_img=img_save_dir,
#         scheduler_type=lr_scheduler,
#         save_dir=save_dirs,
#         save_epc=save_epc,
#         start_epoch=start_epoch,
#         tau=cond_prob,
#         balance_ratio=cls_blc,
#         optimizer_state=optimizer_state)







