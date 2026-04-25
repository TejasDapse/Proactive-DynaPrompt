import argparse
import os
import contextlib

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import math
import pdb
import torchvision.utils as vutils

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

# from clip.custom_clip import get_coop
from clip.custom_clip_lessctx import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import torch.nn.functional as F
# pdb.set_trace()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

import json
import os
import time
from collections import defaultdict

import torch
import torchvision.utils as vutils


import torch
import json
from pathlib import Path


# from dynaprompt_logger import DynaPromptExperimentLogger

def select_confident_samples(logits, top):
    """Select the lowest-entropy subset of samples within a batch.

    Computes the per-sample entropy from logits and returns the `top`
    fraction of samples with the lowest entropy. If `logits` has an extra
    leading dimension (e.g., per-prompt or per-view), entropy is averaged
    across that dimension first.

    Args:
        logits (torch.Tensor): Logits with shape (N, C) or possibly
            (N, M, C).
        top (float): Fraction (0..1) of samples to keep (lowest entropy).

    Returns:
        (torch.Tensor, torch.Tensor): (selected_logits, selected_indices)
    """
    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    if batch_entropy.dim() > 1:
        batch_entropy = batch_entropy.mean(-1)
    
    # Bug fix: Ensure at least one sample is selected if top > 0 to avoid empty pools
    num_samples = logits.size()[0]
    k = max(1, int(num_samples * top)) if (top > 0 and num_samples > 0) else 0
    idx = torch.argsort(batch_entropy, descending=False)[:k]
    return logits[idx], idx

def avg_entropy(outputs):
    """Compute entropy of the average predictive distribution.

    Given a set of model outputs (logits) for multiple augmentations or
    views, this function computes the log-probabilities per-sample,
    aggregates them into an average log-probability distribution across the
    augmentations (in log-space for numerical stability), and returns the
    entropy of that averaged distribution.

    Args:
        outputs (torch.Tensor): Logits with shape (N_views, num_classes).

    Returns:
        torch.Tensor: Scalar entropy value for the averaged prediction.
    """
    if outputs.numel() == 0:
        return torch.tensor(0.0).to(outputs.device)
    
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, C]
    # Use math.log to avoid RuntimeWarning: divide by zero
    avg_logits = logits.logsumexp(dim=0) - math.log(logits.shape[0]) # log-average across views
    
    # Stable entropy calculation: H = -sum(p * log_p)
    probs = torch.exp(avg_logits)
    ent = -(avg_logits * probs)
    ent = torch.nan_to_num(ent, nan=0.0) # Handle p=0 cases (0 * -inf = 0)
    return ent.sum(dim=-1)

def avgweighted_entropy(outputs):
    """Placeholder for weighted average entropy computation.

    This function currently mirrors `avg_entropy` but is intended for
    implementations where per-view logits may be weighted when averaging.
    A debugger `pdb.set_trace()` call exists for interactive debugging; it
    should be removed or guarded in production runs.

    Args:
        outputs (torch.Tensor): Logits tensor across views.

    Returns:
        torch.Tensor: Entropy of the (weighted) averaged distribution.
    """
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, C]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # log-average across views
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    pdb.set_trace()
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_avg_entropy(logits, top):
    """Select low-entropy items per-sample when inputs have extra dims.

    If `logits` has shape (num_views, batch, num_classes) this function
    permutes to (batch, num_views, num_classes) and computes per-view
    entropies, selects the lowest-entropy subset (fraction `top`) per
    sample and returns the mean of those selected entropies per sample.

    Args:
        logits (torch.Tensor): Tensor of shape (V, B, C) or similar.
        top (float): Fraction to select among the `V` views.

    Returns:
        torch.Tensor: Mean selected entropies per batch element.
    """
    if len(logits.size()) == 3:
        logits = logits.permute(1, 0, 2)
        batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
        s_entropys = batch_entropy.topk(int(logits.size()[1] * top), dim=-1, largest=False)[0].mean(-1)

        return s_entropys

def select_avg_entropy_pred(logits, top):
    """Return mean entropy of the lowest-entropy subset and corresponding logits.

    For a batch of logits (shape N x C), compute per-sample entropies,
    select the lowest `top` fraction, and return both the mean selected
    entropies and the subset of logits corresponding to the selected
    indices.

    Args:
        logits (torch.Tensor): Tensor of shape (N, C).
        top (float): Fraction of samples to select (0..1).

    Returns:
        (torch.Tensor, torch.Tensor): (mean_selected_entropies, selected_logits)
    """
    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    # pdb.set_trace()
    s_entropys = batch_entropy.topk(int(logits.size()[0] * top), dim=-1, largest=False)[0].mean(-1)
    output = logits[batch_entropy.topk(int(logits.size()[0] * top), dim=-1, largest=False)[1]]

    return s_entropys, output

def entropy(outputs):
    """Compute entropy from logits using log-space numerics.

    Converts logits to log-probabilities, then computes entropy as
    -sum(p * log p) in a numerically stable fashion.

    Args:
        outputs (torch.Tensor): Logits tensor (..., num_classes).

    Returns:
        torch.Tensor: Entropy per leading dimension of `outputs`.
    """
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.

    This computes the entropy directly via softmax probabilities and the
    corresponding log-softmax for stability.
    """
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)

def test_time_tuning(model, inputs, optimizer, scaler, args, log_file, device, use_cuda, candidate_prompt_indices=None):
    """Run test-time prompt tuning/adaptation for a single input.

    Performs `args.tta_steps` iterations of adaptation. For CoCoOp
    (args.cocoop == True) the function expects `inputs` to be a tuple
    `(image_feature, pgen_ctx)` and optimizes `pgen_ctx`. For standard
    CoOp it adapts the prompt learner parameters using an entropy-based
    objective computed over a selected subset of augmentations.

    The function logs diagnostics using `logger` and `log_file` and
    supports mixed-precision training when `use_cuda` is True and a
    `scaler` is provided.

    Returns:
        For CoCoOp: adapted prompt context tensor.
        Otherwise: (selected_prompt_indices, raw_pred, raw_entropy, conf)
    """
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    loss = torch.tensor(0.0).to(device)
    selection_empty = False  # Default: at least one prompt is always selected
    exactnumber = np.array([0])
    raw_pred, raw_entropy, conf = None, None, None
    # Modern AMP handles both CUDA and MPS
    device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
    autocast_context = torch.amp.autocast(device_type=device_type) if device_type != "cpu" else contextlib.nullcontext()
    for j in range(args.tta_steps):
        with autocast_context:
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                # Pass candidate_prompt_indices to model so only Top-K prompts
                # are encoded by the text encoder during TTA — NOT all N buffer prompts.
                # This is the key fix: encoding all N prompts during backprop causes OOM
                # as the buffer grows unboundedly with --unlimited_buffer.
                if candidate_prompt_indices is not None:
                    if not torch.is_tensor(candidate_prompt_indices):
                        candidate_prompt_indices = torch.tensor(candidate_prompt_indices, device=device, dtype=torch.long)
                    # +1 because model uses 1-indexed prompt selection
                    output = model(inputs, candidate_prompt_indices + 1)
                else:
                    output = model(inputs)

            raw_pred = output[0].detach()
            if args.num_prompts > 1 and (args.onlinetpt or candidate_prompt_indices is not None):
                # Determine active prompt count for selection
                total_num_p = model.prompt_learner.num_p
                selection_empty = False

                if candidate_prompt_indices is not None:
                    num_active = len(candidate_prompt_indices)
                    # output is already sliced to Top-K prompts: shape [batch, n_k_prompts*n_cls]
                    p_pred = output.view(output.size()[0], num_active, -1).detach()
                else:
                    num_active = args.num_prompts
                    p_pred = output.view(output.size()[0], args.num_prompts, -1).detach()

                view_conf = p_pred.softmax(-1).max(-1)[0].mean(1)
                # logger.log_confidence(view_conf.cpu())

                # view_conf = p_pred.softmax(-1).max(-1)[0]  # (Np, Nv)
                # print("Shape of view_conf:", view_conf.shape)

                # print(f"View confidence: {view_conf.cpu()}")

                # print("-"*100)
                # print(view_conf[0].cpu().tolist())

                # for i in range(view_conf.size(0)):
                #     view_conf_i = view_conf[i].softmax(-1).max().item()
                #     print(f"View {i} confidence: {view_conf_i}")
                #     logger.log_confidence({f"view_{i}_conf": view_conf_i})
                # logger.log_confidence({"per_view_conf": view_conf.cpu().tolist()})


                p_ent = entropy(p_pred).mean(0)
                entropy_metrics = {
                    int(idx): {"entropy": float(val)}
                    for idx, val in enumerate(p_ent.detach().cpu())
                }
                entropy_all_views = entropy(p_pred).detach().cpu().numpy()
                entropy_map = {
                    str(view_idx) + "_" + str(class_idx): float(entropy_all_views[view_idx, class_idx])
                    for view_idx in range(entropy_all_views.shape[0])
                    for class_idx in range(entropy_all_views.shape[1])
                }
                # logger.log_entropy_all_views(entropy_map)

                # logger.log_entropy_selection(entropy_metrics)

                plpd = p_pred[0].max(-1)[0].unsqueeze(0) - p_pred[1:].max(dim=-1)[0]

                plpd_metrics = {
                    int(idx): {"plpd": float(v.detach().cpu().item())}
                    for idx, v in enumerate(plpd.view(-1))
                }

                # logger.log_prob_diff_selection(plpd_metrics)
                plpd = plpd.mean(0)


                log_string(log_file, str(model.prompt_learner.ctx_order))
                ent_order = p_ent.topk(num_active)[1]

                # Find reference prompt position among active prompts
                if candidate_prompt_indices is not None:
                    candidate_list = candidate_prompt_indices.tolist()
                    ref_original = None
                    for p_idx in model.prompt_learner.ctx_order:
                        if p_idx in candidate_list:
                            ref_original = p_idx
                            break
                    if ref_original is None:
                        ref_original = candidate_list[0]
                    init_p_position = candidate_list.index(ref_original)
                else:
                    init_p_position = model.prompt_learner.ctx_order[0]

                exactnumber = torch.where(ent_order==init_p_position)[0].item()
                # logger.log_final_selected_prompts(
                #     exactnumber.tolist() if torch.is_tensor(exactnumber) else exactnumber
                # )

                plpd_order = plpd.topk(num_active)[1]
                exactnumberlist1 = ent_order[min(exactnumber + 1, num_active - 1):]
                exactnumberlist2 = plpd_order[:exactnumber]
                alllist = np.intersect1d(exactnumberlist1.cpu().numpy(), exactnumberlist2.cpu().numpy())

                # Track whether intersection yielded any prompts
                selection_empty = not bool(set(alllist))

                if set(alllist):
                    exactnumber = torch.cat([ent_order[alllist], ent_order[[exactnumber]]], dim=0).cpu().numpy() # with init prompt
                else:
                    exactnumber = ent_order[exactnumber].cpu().numpy()

                # Map candidate-relative indices back to original prompt indices
                if candidate_prompt_indices is not None:
                    exactnumber_rel = exactnumber  # 0..num_active-1 (for raw_pred)
                    exactnumber_mapped = candidate_prompt_indices[exactnumber].cpu().numpy()
                    # output has shape [batch * num_active * n_cls] — use num_active, NOT total_num_p
                    output = output.view(args.batch_size, num_active, -1)[:, exactnumber_rel]
                    exactnumber = exactnumber_mapped  # buffer-level indices for ctx_order
                else:
                    output = output.view(args.batch_size, args.num_prompts, -1)[:, exactnumber]

                exactnumber_list = np.array(exactnumber).reshape(-1).tolist()
                # logger.log_prompt_buffer(model.prompt_learner.ctx_order.copy())
                for i in exactnumber_list:
                    model.prompt_learner.ctx_order.remove(i)
                    model.prompt_learner.ctx_use[i] += 1
                    model.prompt_learner.ctx_order.append(i)

                # logger.log_updated_prompts(model.prompt_learner.ctx_order.copy())

                if candidate_prompt_indices is not None:
                    # raw_pred has [num_active * n_cls] elements — index with candidate-relative positions
                    raw_pred = raw_pred.view(num_active, -1)[exactnumber_rel].mean(0)
                else:
                    raw_pred = raw_pred.view(args.num_prompts, -1)[exactnumber].mean(0)

            else:
                p_ent = entropy(raw_pred)




            raw_entropy = avg_entropy(raw_pred.unsqueeze(0))
            conf = raw_pred.softmax(-1).max().item()

            # print("raw_conf", conf)
            # print("type of raw_conf", type(conf))
            # logger.log_confidence({"raw_conf": conf})

            # view_conf = p_pred.softmax(-1).max(-1)[0].mean(1)
            # logger.log_confidence(view_conf.cpu())
            print(f"Step {j+1}/{args.tta_steps}, Raw entropy: {raw_entropy.item():.4f}, Raw pred: {raw_pred.argmax(-1).item()}")
            output, selected_idx = select_confident_samples(output, args.selection_p)
            # logger.log_threshold(args.selection_p)
            # logger.log_filtered_count(len(selected_idx))
            print(f"Selected {output} samples with lowest entropy for adaptation")
            loss = avg_entropy(output).mean()
            print(f"Adaptation loss (avg entropy of selected samples): {loss.item():.4f}")

        optimizer.zero_grad()
        # compute gradient and do SGD step
        if scaler is not None:
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss = torch.tensor(0.0).to(device)

    if args.cocoop:
        return pgen_ctx

    return exactnumber, raw_pred, raw_entropy, conf, selection_empty

def log_string(log_file, string):
    log_file.write(string + '\n')
    log_file.flush()

def main():
    print("Starting main function...")
    args = parser.parse_args()
    
    # Ensure run_dir is unique to avoid erasing previous results
    if os.path.exists(args.run_dir):
        base_dir = args.run_dir.rstrip('/')
        counter = 1
        new_run_dir = f"{base_dir}_{counter}"
        while os.path.exists(new_run_dir):
            counter += 1
            new_run_dir = f"{base_dir}_{counter}"
        args.run_dir = new_run_dir
    
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"Results will be saved to: {args.run_dir}")
    
    set_random_seed(args.seed)
    print(f"Arguments parsed: test_sets={args.test_sets}, arch={args.arch}, gpu={args.gpu}")
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    """Main evaluation worker: orchestrates model setup, dataset loading, and test-time adaptation.

    This function serves as the orchestrator for the entire evaluation pipeline. It:
    1. Selects and configures the computation device (CUDA, MPS, or CPU).
    2. Constructs either a CoOp or CoCoOp prompt-learning model with class names
       from the specified dataset(s).
    3. Loads pre-trained weights if provided and freezes non-trainable parameters.
    4. Prepares optimizer and mixed-precision (AMP) scaler for test-time adaptation.
    5. Iterates through one or more test datasets, applying test-time adaptation to
       each sample via `test_time_adapt_eval` and logging results.
    6. Prints and writes to log file a summary of top-1 and top-5 accuracies across
       all evaluated datasets.

    The function supports:
    - Multiple datasets (split by "/" in args.test_sets, e.g., "A/R/V/K/I").
    - Two model types: CoOp (trainable prompts) and CoCoOp (image-conditioned context).
    - Test-time augmentation with configurable selection percentile.
    - Both CUDA and non-CUDA (MPS/CPU) backends.

    Args:
        gpu (int): GPU device ID to use. Passed as --gpu argument.
        args (argparse.Namespace): Parsed command-line arguments, including:
            - arch (str): CLIP backbone architecture (e.g., 'ViT-B/16', 'RN50').
            - test_sets (str): Dataset(s) to evaluate, slash-separated (e.g., 'A/R/V').
            - n_ctx (int): Number of learnable context tokens in prompt.
            - ctx_init (str): Initial context words (e.g., 'a_photo_of_a') or None for random.
            - cocoop (bool): If True, use CoCoOp; else use CoOp.
            - load (str): Path to pre-trained model weights or None.
            - tpt (bool): Enable test-time prompt tuning with augmentation.
            - tta_steps (int): Number of adaptation steps per sample.
            - selection_p (float): Percentile of samples to select for adaptation.
            - lr (float): Learning rate for prompt adaptation.
            - seed (int): Random seed for reproducibility.
            - log_date (str): Date/identifier for log file naming.
            - onlinetpt (bool): Enable online prompt tuning with multi-prompt selection.
            - num_prompts (int): Number of prompts to maintain and rotate.
            - batch_size (int): Batch size for evaluation.
            - data (str): Path to dataset root directory.
            - dataset_mode (str): Dataset split to use ('train', 'val', 'test').
            - resolution (int): CLIP image resolution (typically 224).
            - print_freq (int): Frequency of progress output.

    Returns:
        None. Results are printed to stdout and logged to files in the logs/ directory.

    Side effects:
        - Creates and populates log files in logs/ or logs_dynaprompt/.
        - Creates per-sample log directories and JSON metadata in runs_imagenet_dynaprompt/.
        - Modifies args.gpu in-place to reflect the selected GPU.
    """
    print(f"Starting main_worker with gpu={gpu}")
    args.gpu = gpu
    set_random_seed(args.seed)
    
    # Determine device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = 'mps'
        use_cuda = False
    else:
        device = 'cpu'
        use_cuda = False
    
    print(f"Use device: {device}")
    
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    
    print("Creating model...") # (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    print(f"Number of classes for {args.test_sets}: {len(classnames)}")

    if args.cocoop:
        print("Using CoCoOp model")
        model = get_cocoop(args.arch, args.test_sets, device, args.n_ctx)
        assert args.load is not None
        print(f"Loading model weights from {args.load}")
        load_model_weight(args.load, model, device, args) # to load to device
        model_state = deepcopy(model.state_dict())
    else:
        print("Using CoOp model")
        model = get_coop(args.arch, args.test_sets, device, args.n_ctx, args.ctx_init, False, args.num_prompts)
        if args.load is not None:
            print(f"Use pre-trained soft prompt (CoOp) as initialization from {args.load}")
            pretrained_ctx = torch.load(args.load, map_location=device)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx.to(device))
                model.prompt_learner[0].ctx_init_state = pretrained_ctx.to(device)

            prompts = model.prompt_learner()
            print(f"Initialized prompts: {prompts.cpu()}")
        model_state = None

    print("Freezing parameters...")
    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    model = model.to(device)

    # define optimizer
    print("Setting up optimizer...")
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    print("Setting up scaler...")
    if "cuda" in str(device):
        scaler = torch.amp.GradScaler("cuda", init_scale=1000)
        print('=> Using native Torch AMP (CUDA). Training in mixed precision.')
    elif "mps" in str(device):
        # MPS doesn't always require GradScaler, but torch.amp.GradScaler supports it in newer versions
        scaler = None 
        print('=> Using MPS (Metal) backend. GradScaler disabled.')
    else:
        scaler = None
        print('=> Not using AMP.')
    
    # Initialize logger within main_worker after args are processed
    # global logger
    # logger = DynaPromptExperimentLogger(args.run_dir)

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    print(f"Evaluating datasets: {datasets}")
    for set_id in datasets:
        print(f"Processing dataset: {set_id}")
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1,
                                            augmix=len(set_id)>1)
            # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=int(args.batch_size / args.loss_update_step) - 1,
            #                                  augmix=len(set_id) > 1) ### update model for each loss_update_step samples
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        if args.arch == 'ViT-B/16':
            model_arch = 'ViT-B-16'
        elif args.arch == 'RN50':
            model_arch = 'RN50'
        elif args.arch == 'ViT-B/32':
            model_arch = 'ViT-B-32'
        
        # Consolidation: Put all logs inside the run_dir
        log_dir = os.path.join(args.run_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        prefix = 'otpt_' if args.onlinetpt else ''
        logname = os.path.join(log_dir, "{}{}_s{}_{}ctx_{}_{}_nump{}_{}%augs_{}ctxinit{}.txt".format(prefix, args.log_date, args.seed, args.n_ctx, model_arch, set_id, args.num_prompts, args.selection_p, args.lr, args.ctx_init))

        log_file = open(logname, 'w')
        print("evaluating: {}".format(set_id))
        log_string(log_file, "evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        # pdb.set_trace()
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.to(device)
        else:
            model.reset_classnames(classnames, args.arch)

        print(f"Building dataset for {set_id}")
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        
        # Sampling dataset for quick testing
        if args.sample_size > 0 and len(val_dataset) > args.sample_size:
            print(f"Sampling dataset: using {args.sample_size} samples out of {len(val_dataset)}")

            from torch.utils.data import Subset

            # 1. Define your desired sample size
            sample_size = min(args.sample_size, len(val_dataset)) 
            
            # 2. Generate random indices
            # torch.randperm(n) creates a shuffled list of indices from 0 to n-1
            indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:sample_size].tolist()

            # 3. Create the random subset
            val_dataset = Subset(val_dataset, indices)

            # val_dataset = torch.utils.data.Subset(val_dataset, range(args.sample_size))
            
        print("number of test samples: {}".format(len(val_dataset)))
        log_string(log_file, "number of test samples: {}".format(len(val_dataset)))
        print(f"Log file: {logname}")
        print(f"Creating DataLoader with batch_size={batchsize}")
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=0, pin_memory=False)
            
        print(f"Starting test_time_adapt_eval for {set_id}")
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, log_file, args, device, use_cuda)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
            log_string(log_file, "=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
            log_string(log_file, "=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("All datasets evaluated. Printing summary...")
    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")

    log_string(log_file, "======== Result Summary ========")
    log_string(log_file, "params: nstep	lr	bs")
    log_string(log_file, "params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    log_string(log_file, "\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        log_string(log_file, "{}".format(id))
    log_string(log_file, "\n")
    for id in results.keys():
        log_string(log_file, "{:.2f}".format(results[id][0]))
    log_string(log_file, "\n")

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, log_file, args, device, use_cuda):
    """Evaluate model on a test set with per-sample test-time adaptation.

    This function iterates through a validation dataset, performing test-time
    adaptation (TTA) for each sample and computing top-1 and top-5 accuracies.
    For each sample, it:
    1. Resets the model's learnable prompts to their initial state (if not CoCoOp).
    2. Runs `test_time_tuning` to adapt prompts using entropy minimization on
       augmented views.
    3. Performs final inference on the original and augmented inputs.
    4. Computes accuracy and logs predictions, metrics, and augmentations.

    The function leverages the global `logger` (DynaPromptExperimentLogger) to
    record per-sample diagnostics including:
    - Original images and their augmentations.
    - Confidence scores and entropy metrics (raw and adapted).
    - Selected prompts and their ordering.
    - Predictions (raw and post-adaptation) with accuracies.

    Key behaviors:
    - For CoCoOp models: computes image features once and optimizes context
      conditioned on those features.
    - For CoOp models: adapts shared prompt embeddings over TTA steps.
    - Supports online multi-prompt tuning where the "best" prompts are rotated
      to the front of a queue (if args.onlinetpt and args.num_prompts > 1).
    - Uses mixed-precision training if `scaler` is provided and `use_cuda` is True.
    - Tracks timing and accuracy metrics using AverageMeter utilities.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader providing (images, target)
            tuples. Images may be a list (for multi-view augmentation) or a tensor.
        model: The prompt-learning CLIP model (CoOp or CoCoOp) to adapt and evaluate.
        model_state (dict or None): Saved model state (used for CoCoOp; can be None).
        optimizer (torch.optim.Optimizer or None): Optimizer for CoOp prompt updates.
            None for CoCoOp (optimizer is created inside test_time_tuning).
        optim_state (dict or None): Saved optimizer state to reset between samples.
        scaler (torch.cuda.amp.GradScaler or None): Mixed-precision loss scaler.
            None if not using AMP (e.g., on CPU or MPS devices).
        log_file: Open file handle for writing per-batch log messages.
        args (argparse.Namespace): Configuration namespace with fields:
            - tta_steps (int): Number of adaptation iterations per sample.
            - selection_p (float): Fraction of augmentations to select (lowest entropy).
            - tpt (bool): Whether test-time augmentation is enabled.
            - cocoop (bool): If True, model is CoCoOp; else CoOp.
            - onlinetpt (bool): Enable online multi-prompt selection.
            - num_prompts (int): Number of prompts (if onlinetpt).
            - print_freq (int): Print progress every N batches.
            - batch_size (int): Batch size (used in prompt reshape logic).
            - lr (float): Learning rate (passed to test_time_tuning).
            - log_date (str): Date identifier for conditional logic.
        device (str): Computation device ('cuda:X', 'mps', or 'cpu').
        use_cuda (bool): True if using CUDA (enables AMP autocast context).

    Returns:
        list: [top1_avg, top5_avg] - Average top-1 and top-5 accuracies over
            all samples in val_loader.

    Side effects:
        - Logs per-sample data via `logger` (images, metrics, predictions).
        - Writes progress to `log_file`.
        - Modifies model state (prompt parameters) during adaptation and resets
          between samples if CoOp is used.
        - Updates AverageMeter objects (batch_time, top1, top5) used for summary stats.
    """
    print(f"Starting test_time_adapt_eval with {len(val_loader)} batches")
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    print("Model set to eval mode")
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
            print("Model prompts reset")
    
    if args.proactive_routing:
        # Reset counters for the new dataset
        model.routing_stats["dropped_samples"] = 0
        model.routing_stats["false_hits"] = 0
        model.routing_stats["hits"] = 0
        model.routing_stats["misses"] = 0
        model.routing_stats["total_samples"] = 0
    
    total_start_time = time.time()
    end = time.time()
    print("Starting batch processing...")
    for i, (images, target) in enumerate(val_loader):
        # pdb.set_trace()
        # print(i)
        # logger.start_sample(i)
        if isinstance(images, list):
            original_image = images[0][0]   # first view, first sample
        else:
            original_image = images[0]

        # logger.start_sample(
        #     image_tensor=original_image,
        #     label=target.item(),
        #     image_name=f"sample_{i}"
        # )

        print(f"Processing batch {i+1}/{len(val_loader)}")
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=use_cuda)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=use_cuda)
            image = images
        target = target.to(device, non_blocking=use_cuda)
        images = torch.cat(images, dim=0)
        print(f"Data moved to {device}")

        # logger.log_augmentations(images)





        # pdb.set_trace()
        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            # if args.tta_steps > 0:
            if args.tta_steps > 0 and not args.onlinetpt:
                with torch.no_grad():
                    # print('___________Prompts are reset___________')
                    model.reset()
                    print("Prompts reset for TTA")
            elif args.onlinetpt and args.num_prompts > 1:
                with torch.no_grad():
                    if model.prompt_learner.ctx_use.count(0) == 0 and 'noapp' not in args.log_date:
                        time0 = time.time()
                        log_string(log_file, "All prompts are used, reset prompt {}".format(model.prompt_learner.ctx_order[0]))
                        model.prompt_learner.ctx_use[model.prompt_learner.ctx_order[0]] = 0
                        model.prompt_learner.ctx[model.prompt_learner.ctx_order[0]] = model.prompt_learner.ctx[model.prompt_learner.ctx_order[0]] * 0

            optimizer.load_state_dict(optim_state)
            
            # Save state for possible restoration if adaptation criteria are not met
            saved_ctx = model.prompt_learner.ctx.data.clone()
            saved_num_p = model.prompt_learner.num_p
            saved_ctx_init_state = model.prompt_learner.ctx_init_state.detach().clone()
            saved_ctx_order = model.prompt_learner.ctx_order.copy()
            saved_ctx_use = model.prompt_learner.ctx_use.copy()

            # Proactive Routing (Optional Component)
            p_s = None
            raw_pred, raw_ent, raw_conf = None, None, None
            hit = False
            was_false_hit = False
            selection_empty = False
            
            if args.proactive_routing:
                # 1. Visual Feature Extraction
                # Use the original image (first view)
                with torch.no_grad():
                    z_x = model.get_visual_features(image[0:1]) # original_image is image[0]
                
                # 2. Proactive Routing Decision
                if len(model.prototypes) > 0:
                    # Calculate cosine similarity with all existing prototypes dynamically
                    keys = list(model.prototypes.keys())
                    protos = torch.stack([model.prototypes[k] for k in keys]) # (N, embed_dim)
                    sims = torch.matmul(protos, z_x.T).squeeze(1) # (N,)
                    
                    similarities = torch.full((model.prompt_learner.num_p, model.prompt_learner.n_cls), -1.0, device=z_x.device, dtype=z_x.dtype)
                    p_indices = [k[0] for k in keys]
                    c_indices = [k[1] for k in keys]
                    similarities[p_indices, c_indices] = sims
                    
                    # For each prompt, take the maximum similarity across its classes
                    max_sims_per_prompt, _ = similarities.max(dim=1)
                    max_sim, k_star = max_sims_per_prompt.max(dim=0)
                    
                    if max_sim.item() > args.routing_threshold:
                        hit = True
                        p_s = k_star.view(1) # Ensure it's a tensor
                        print(f"Proactive Routing Hit! Prompt index: {p_s.item()}, Similarity: {max_sim.item():.4f}")
                        
                        device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
                        autocast_ctx = torch.amp.autocast(device_type=device_type) if device_type != "cpu" else contextlib.nullcontext()
                        
                        # Calculate Raw Entropy BEFORE adaptation
                        with torch.no_grad():
                            with autocast_ctx:
                                init_output = model(image[0:1], p_s + 1)
                                raw_pred = init_output.detach().mean(0)
                        raw_ent = avg_entropy(raw_pred.unsqueeze(0))
                        raw_conf = raw_pred.softmax(-1).max().item()

                        # Use the current sample views to fine-tune the matched prompt
                        print(f"Fine-tuning Hit prompt index {p_s.item()} on current sample...")
                        for tta_j in range(args.tta_steps):
                            with autocast_ctx:
                                # Tune the specific prompt identified by k_star
                                hit_output = model(images, p_s + 1)
                                hit_output_sel, _ = select_confident_samples(hit_output, args.selection_p)
                                hit_loss = avg_entropy(hit_output_sel).mean()
                            
                            optimizer.zero_grad()
                            if scaler is not None:
                                scaler.scale(hit_loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                hit_loss.backward()
                                optimizer.step()

                        # Post-adaptation check for False Hit
                        with torch.no_grad():
                            with autocast_ctx:
                                # Final output for this sample from the adapted hit prompt
                                post_output = model(image[0:1], p_s + 1)
                                post_pred = post_output.detach().mean(0)
                        post_ent = avg_entropy(post_pred.unsqueeze(0)).item()
                        
                        if post_ent > args.fallback_threshold:
                            print(f"False Hit! Post-adaptation entropy {post_ent:.4f} > fallback threshold {args.fallback_threshold}. Falling back to Discovery.")
                            hit = False
                            was_false_hit = True
                            model.routing_stats["false_hits"] += 1
                            # Restoration to pre-hit state
                            model.prompt_learner.ctx.data.copy_(saved_ctx)
                            model.prompt_learner.num_p = saved_num_p
                            model.prompt_learner.ctx_order = saved_ctx_order.copy()
                            model.prompt_learner.ctx_use = saved_ctx_use.copy()
                            optimizer.load_state_dict(optim_state)
                        else:
                            print(f"Hit confirmed. Post-adaptation entropy: {post_ent:.4f}")
            
            if not hit:
                if args.proactive_routing:
                    print("Proactive Routing Miss/New Domain.")
                    model.routing_stats["misses"] += 1
                
                # Case B (Miss) or routing disabled: perform DynaPrompt Selection
                print("Running test_time_tuning (Augmentation + Optimization)")

                if not was_false_hit:
                    # Compute Top-K candidate prompts from similarity (proactive routing only)
                    top_k_candidates = None
                    if args.proactive_routing and len(model.prototypes) > 0:
                        # max_sims_per_prompt was computed in Step 2 above
                        k = min(args.top_k_prompts, len(max_sims_per_prompt))
                        top_k_candidates = max_sims_per_prompt.topk(k)[1]
                        print(f"Top-{k} candidate prompts by similarity: {top_k_candidates.tolist()}")

                    p_s, raw_pred, raw_ent, raw_conf, selection_empty = test_time_tuning(
                        model, images, optimizer, scaler, args, log_file, device, use_cuda,
                        candidate_prompt_indices=top_k_candidates
                    )
                else:
                    # For false hits, we bypass selection and force discovery
                    selection_empty = True
                    print("Bypassing DynaPrompt Selection after False Hit.")

                # Handle 0-prompt sub-case: specialize a prompt for the new domain
                if args.proactive_routing and selection_empty:
                    print("Selection empty: discovering new domain and specializing expert.")

                    # Identify the target index and initialize to base prompt delta
                    with torch.no_grad():
                        base_ctx = model.prompt_learner.ctx_init_state[0:1].detach().clone()
                        
                        if args.unlimited_buffer:
                            # Case 5: PLR with Unlimited Buffer Growth
                            target_idx = model.prompt_learner.add_prompt(base_ctx[0])
                            model.expand_prototype_bank()
                            print(f"Discovery: Added new prompt at index {target_idx}. Buffer size: {model.prompt_learner.num_p}")
                        else:
                            # Case 4: PLR with Fixed Buffer Size (Replacement Strategy)
                            target_idx = model.prompt_learner.ctx_order[0]
                            print(f"Discovery: Buffer full. Replacing oldest expert at index {target_idx}.")
                            # Overwrite existing index
                            model.prompt_learner.ctx[target_idx].copy_(base_ctx[0])
                            model.prompt_learner.ctx_init_state[target_idx].copy_(base_ctx[0])
                            model.prompt_learner.ctx_use[target_idx] = 1

                    # Rebuild optimizer for the discovery pass (essential for new parameters/delta updates)
                    trainable_param = model.prompt_learner.parameters()
                    optimizer = torch.optim.AdamW(trainable_param, args.lr)
                    optim_state = deepcopy(optimizer.state_dict())

                    # Direct single-prompt TTA on the target index
                    # We bypass test_time_tuning() because its multi-prompt selection logic
                    # (PLPD, entropy ordering) crashes with only 1 candidate prompt.
                    device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
                    autocast_ctx = torch.amp.autocast(device_type=device_type) if device_type != "cpu" else contextlib.nullcontext()
                    target_p_s = torch.tensor([target_idx], device=device, dtype=torch.long)
                    
                    # Calculate Raw Entropy BEFORE adaptation for Discovery
                    with torch.no_grad():
                        with autocast_ctx:
                            init_output = model(image[0:1], target_p_s + 1)
                            raw_pred = init_output.detach().mean(0)
                    raw_ent = avg_entropy(raw_pred.unsqueeze(0))
                    raw_conf = raw_pred.softmax(-1).max().item()

                    for tta_j in range(args.tta_steps):
                        with autocast_ctx:
                            disc_output = model(images, target_p_s + 1)  # +1 because model uses 1-indexed prompt selection
                            disc_output_sel, _ = select_confident_samples(disc_output, args.selection_p)
                            disc_loss = avg_entropy(disc_output_sel).mean()
                        
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.scale(disc_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            disc_loss.backward()
                            optimizer.step()

                    # Prompt and Index tracking
                    p_s = np.array([target_idx])

                update_idx = p_s.reshape(-1)[0].item()
            else:
                model.routing_stats["hits"] += 1
                update_idx = p_s.item()

            print("Test-time process completed")
            # print(f"Selected prompt index: {p_s}, Raw prediction: {raw_pred.argmax(-1).item() if raw_pred is not None else 'N/A'}, Raw entropy: {raw_ent.item() if raw_ent is not None else 'N/A'}")
        else:
            with torch.no_grad():
                autocast_context_inf = torch.cuda.amp.autocast() if use_cuda else contextlib.nullcontext()
                with autocast_context_inf:
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            print("Running test_time_tuning for CoCoOp")
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args, log_file, device, use_cuda)
            print("Test-time tuning for CoCoOp completed")
            print(f"Selected prompt context: {pgen_ctx}")

        print("Running final inference")
        ########################################################################################################
        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
            autocast_context_inf = torch.amp.autocast(device_type=device_type) if device_type != "cpu" else contextlib.nullcontext()
            with autocast_context_inf:
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    p_s = p_s.reshape(-1)
                    output = model(image, p_s + 1)
                    output = output.view(output.size()[0], p_s.reshape(-1).shape[0], -1)
                    output = output.mean(1)

                    ent = avg_entropy(output.unsqueeze(0))

            # --- Conditional Buffer and Prototype Update ---
            if not args.cocoop:
                adapted_entropy = ent.item() if 'ent' in locals() and ent is not None else 0.0
                raw_entropy_val = raw_ent.item() if raw_ent is not None else 0.0
                # Condition 1: Adapted entropy < threshold
                # Condition 2: Improved entropy (Raw - Adapted >= 0)
                # Note: User specified "difference between Adapted and Raw >= 0". 
                # Interpreting as "absolute difference" or "improvement" usually depends on context.
                # Here we assume it means it should not get worse: raw_entropy_val - adapted_entropy >= 0
                if adapted_entropy < args.entropy_threshold and (raw_entropy_val - adapted_entropy) >= 0:
                    if args.proactive_routing:
                        model.routing_stats["total_samples"] += 1
                        image_id = f"sample_{i}"
                        if update_idx not in model.routing_stats["prompt_usage"]:
                            model.routing_stats["prompt_usage"][update_idx] = {"count": 0, "image_ids": []}
                        model.routing_stats["prompt_usage"][update_idx]["count"] += 1
                        model.routing_stats["prompt_usage"][update_idx]["image_ids"].append(image_id)

                        pred_cls = raw_pred.argmax(-1).item() if raw_pred is not None else 0
                        n = model.prototypes_count.get((update_idx, pred_cls), 0)
                        if n == 0:
                            model.prototypes[(update_idx, pred_cls)] = z_x.view(-1).detach().clone()
                        else:
                            old_proto = model.prototypes[(update_idx, pred_cls)]
                            model.prototypes[(update_idx, pred_cls)] = (old_proto * n + z_x.view(-1).detach().clone()) / (n + 1)
                        model.prototypes_count[(update_idx, pred_cls)] = n + 1
                        
                        if not hit:
                            print(f"Updated prototype for prompt index {update_idx}, class {pred_cls} (Cumulative Average)")
                        else:
                            print(f"Refined prototype for prompt index {update_idx}, class {pred_cls} after Hit")
                    print(f"Sample {i}: Criteria met (Ent: {adapted_entropy:.4f} < {args.entropy_threshold}, Diff: {raw_entropy_val - adapted_entropy:.4f} >= 0). Prompt updated.")
                else:
                    # Restore state
                    with torch.no_grad():
                        model.prompt_learner.ctx = nn.Parameter(saved_ctx)
                        model.prompt_learner.num_p = saved_num_p
                        model.prompt_learner.ctx_init_state = saved_ctx_init_state
                        model.prompt_learner.ctx_order = saved_ctx_order
                        model.prompt_learner.ctx_use = saved_ctx_use
                    
                    model.routing_stats["dropped_samples"] += 1
                    print(f"Sample {i}: Criteria NOT met (Ent: {adapted_entropy:.4f}, Raw: {raw_entropy_val:.4f}). Prompt dropped. Total dropped: {model.routing_stats['dropped_samples']}")

        sample_duration = time.time() - end
        log_string(log_file,
                       "Sample: {}, Time: {:.3f}s, Raw entropy: {:.2f}, Adapted entropy: {:.2f}, Raw pred: {}, Adapted pred: {}, Target: {}".format(
                           i, sample_duration, raw_ent.item(), ent.item(), raw_pred.argmax(-1).item(), output.argmax(-1).item(),
                           target.item()))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        adapted_conf = output.softmax(-1).max().item()
        # logger.log_prediction(
        #     raw_pred=raw_pred.argmax(),
        #     adapted_pred=output.argmax(),
        #     raw_entropy=raw_ent.item(),
        #     adapted_entropy=ent.item(),
        #     raw_conf=raw_conf,
        #     adapted_conf=adapted_conf
        # )
        # logger.end_sample()


        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        
        # Flush MPS memory cache explicitly to prevent high-watermark leaks
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            progress.display(i)
            entries = [progress.prefix + progress.batch_fmtstr.format(i)]
            entries += [str(meter) for meter in progress.meters]
            log_string(log_file, '\t'.join(entries))

    progress.display_summary()
    total_duration = time.time() - total_start_time
    print("=> Total evaluation time: {:.2f}s".format(total_duration))
    log_string(log_file, "=> Total evaluation time: {:.2f}s".format(total_duration))
    
    # Save Routing Stats to JSON at the end of evaluation
    if args.proactive_routing:
        stats_path = os.path.join(os.path.dirname(log_file.name), "routing_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(model.routing_stats, f, indent=4)
        print(f"Saved proactive routing stats to {stats_path}")

    print(f"Evaluation completed. Top1: {top1.avg:.2f}, Top5: {top5.avg:.2f}, Dropped: {model.routing_stats.get('dropped_samples', 0)}, False Hits: {model.routing_stats.get('false_hits', 0)}")
    return [top1.avg, top5.avg]

# Global logger placeholder (will be initialized in main_worker)
# logger = None 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--run_dir', type=str, default='runs/exp', help='base directory to store all logs, results, and experiment artifacts')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_date', type=str, default='')
    parser.add_argument('--sample_size', type=int, default=0, help='number of samples to test (0 for all)')
    parser.add_argument('--onlinetpt', action='store_true', default=False, help='run online prompt tuning')
    parser.add_argument('--num_prompts', default=1, type=int, help='number of prompts to tune')
    parser.add_argument('--proactive_routing', action='store_true', default=False, help='enable proactive prompt routing based on visual similarity')
    parser.add_argument('--routing_threshold', default=0.7, type=float, help='threshold for proactive routing hit')
    parser.add_argument('--top_k_prompts', default=3, type=int, help='number of top prompts to consider during miss recovery (proactive routing)')
    parser.add_argument('--unlimited_buffer', action='store_true', default=False, help='allow the prompt buffer to grow infinitely on discovery (proactive routing)')
    parser.add_argument('--entropy_threshold', default=0.4, type=float, help='entropy threshold for conditional buffer update (memory addition)')
    parser.add_argument('--fallback_threshold', default=0.4, type=float, help='entropy threshold for triggering discovery fallback on false hits (prediction quality)')

    print("Arguments for the run:")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    main()