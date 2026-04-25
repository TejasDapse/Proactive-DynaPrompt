import pdb

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # x = prompts + self.positional_embedding.type(self.dtype)
        # pdb.set_trace()
        if len(prompts.size()) > 3:
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(prompts.size(0), 1, 1)
            tokenized_prompts = tokenized_prompts.view(-1, tokenized_prompts.size()[-1])
        x = prompts.view(-1, prompts.size()[-2], prompts.size()[-1]) + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=4, ctx_init=None, ctx_position='end', learned_cls=False, num_p=1
                 ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.num_p = num_p
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)
        # pdb.set_trace()

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx0 = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            ctx_vectors = torch.zeros(embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].size(), dtype=dtype)
            self.init_ctx = embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].clone()
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)

        # pdb.set_trace()
        ##### multiple prompts #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(num_p, 1, 1)  # (N, L, D)
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (L, N, D)
        ##### feature initialization #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (N, L, D)

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.ctx_order = list(range(num_p))
        self.ctx_use = [0] * num_p
        # pdb.set_trace()

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        print(f"Initialized prompt tokens: {prompts}")
        with open("initialized_prompts.txt", "w") as f:
            for p in prompts:
                f.write(p + "\n")

            


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # pdb.set_trace()
        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx0 - n_ctx, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx0 :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_ctx0 = n_ctx0
        self.classnames = classnames

    def reset(self):
        # pdb.set_trace()
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def add_prompt(self, tuned_ctx_vector):
        """Dynamically add a new tuned prompt to the buffer (unlimited growth).

        Args:
            tuned_ctx_vector: The tuned prompt context vector, shape (n_ctx, ctx_dim).

        Returns:
            int: The index of the newly added prompt.
        """
        with torch.no_grad():
            new_ctx = tuned_ctx_vector.detach().clone()
            if new_ctx.dim() == 2:
                new_ctx = new_ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)

            # Expand ctx parameter using CPU to avoid MPS backend segmentation fault bugs
            device_orig = self.ctx.device
            cpu_ctx = self.ctx.data.cpu()
            cpu_new_ctx = new_ctx.cpu()
            
            new_ctx_data = torch.cat([cpu_ctx, cpu_new_ctx], dim=0).to(device_orig)
            self.ctx = nn.Parameter(new_ctx_data)
            self.ctx.requires_grad_(True)

            # Update init state (new prompt's init = its current tuned state)
            cpu_init_state = self.ctx_init_state.cpu()
            self.ctx_init_state = torch.cat([cpu_init_state, cpu_new_ctx], dim=0).to(device_orig)

            # Update tracking
            new_idx = self.num_p
            self.num_p += 1
            self.ctx_order.append(new_idx)
            self.ctx_use.append(1)  # Mark as used since we just created it

        return new_idx

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1 + self.n_ctx0 - self.n_ctx, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx0 :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None, prompt_indices=None):
        """Build prompt token sequences.

        Args:
            init: Optional override for context vectors.
            prompt_indices: Optional 0-based LongTensor of which prompt rows to
                build. When supplied only those rows are materialized, keeping
                peak memory at O(k * n_cls) instead of O(num_p * n_cls). This
                is critical for the unlimited-buffer setting where num_p can
                grow large during a single evaluation run.
        """
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx + self.init_ctx

        # --- Determine whether we are in multi-prompt mode ---
        # multi_prompt_mode = True whenever the buffer has >1 prompt AND we
        # want to produce a [num_p, n_cls, seq, dim] output (or a k-subset of it).
        multi_prompt_mode = (self.num_p > 1 and self.batch_size is None)

        # --- Subset selection (memory-critical path) ---
        # Slice ctx BEFORE the expensive n_cls expansion so we only build
        # the rows we actually need during this forward pass.
        if prompt_indices is not None and multi_prompt_mode:
            ctx = ctx[prompt_indices]          # (k, n_ctx, dim)
            effective_num_p = len(prompt_indices)
        else:
            effective_num_p = self.num_p

        # Expand ctx to include the class dimension
        if ctx.dim() == 2:
            # Single-prompt, no multi-prompt mode
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif multi_prompt_mode:
            # Multi-prompt: ctx is (k, n_ctx, dim) → (k, n_cls, n_ctx, dim)
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix   # [n_cls, 1, dim]
        suffix = self.token_suffix   # [n_cls, *, dim]

        if self.batch_size is not None:
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)
        elif multi_prompt_mode:
            # Add prompt dimension: [n_cls, *, dim] → [k, n_cls, *, dim]
            prefix = prefix.unsqueeze(0).repeat(effective_num_p, 1, 1, 1)
            suffix = suffix.unsqueeze(0).repeat(effective_num_p, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [prefix, ctx, cls, suffix],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [prefix, ctx, suffix],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            half_n_ctx = self.split_idx if self.split_idx is not None else self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompts.append(torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1
                ))
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompts.append(torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i], dim=1
                ))
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts



class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, num_p=1):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.num_p = num_p
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, num_p)
        self.criterion = criterion
        
        # Proactive Routing: Prototype Buffer
        # Initializing prototypes with zeros. They will be filled as we encounter new domains.
        embed_dim = clip.visual.output_dim if hasattr(clip.visual, 'output_dim') else clip.ln_final.weight.shape[0]
        n_cls = len(classnames)
        # Dynamically sized dictionary prototypes (saves memory) 
        self.prototypes = {}
        self.prototypes_count = {}

        # Routing Stats for JSON logging
        self.routing_stats = {
            "total_samples": 0,
            "hits": 0,
            "misses": 0,
            "false_hits": 0,
            "dropped_samples": 0,
            "prompt_usage": {} # {prompt_id: {"count": 0, "image_ids": []}}
        }
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_visual_features(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def expand_prototype_bank(self):
        """Expand the prototype bank by one row for a newly added prompt.
        Since prototypes are now dynamically tracked via dictionaries, this is a no-op."""
        pass

    def get_text_features(self, p_s=None):
        """Encode text features for given prompt indices.

        Args:
            p_s: 1-based prompt index tensor/list. Only those prompts are built
                 by PromptLearner, keeping memory O(k * n_cls).
                 If None, ALL prompts are built chunk-by-chunk (used only when
                 full-buffer encoding is truly needed, e.g. final inference).
        """
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        if p_s is not None:
            # Convert 1-based p_s to 0-based indices for PromptLearner
            if not isinstance(p_s, torch.Tensor):
                p_idx = torch.tensor(p_s, dtype=torch.long,
                                     device=self.prompt_learner.ctx.device) - 1
            else:
                p_idx = p_s.long() - 1

            # Build ONLY the requested prompt rows (memory-efficient path)
            prompts = self.prompt_learner(prompt_indices=p_idx)
            t_features = self.text_encoder(prompts, tokenized_prompts)
            return t_features / t_features.norm(dim=-1, keepdim=True)
        else:
            # Full-buffer path: process one prompt at a time to cap peak memory.
            # This is only reached during final inference (no gradients needed
            # there, but we still guard with chunk_size=1 for safety).
            num_prompts = self.prompt_learner.num_p
            all_features = []
            for i in range(num_prompts):
                p_idx_i = torch.tensor([i], dtype=torch.long,
                                       device=self.prompt_learner.ctx.device)
                chunk_prompts = self.prompt_learner(prompt_indices=p_idx_i)
                chunk_feat = self.text_encoder(chunk_prompts, tokenized_prompts)
                all_features.append(chunk_feat / chunk_feat.norm(dim=-1, keepdim=True))
            return torch.cat(all_features, dim=0)

    def inference(self, image, p_s=None):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        self.image_features = image_features.mean(0)

        if self.num_p > 1 and p_s is not None:
            text_features = self.get_text_features(p_s)
        else:
            # No p_s: encode all prompts one-at-a-time (safe even with large buffer)
            text_features = self.get_text_features()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def forward(self, input, p_s=None):
        # pdb.set_trace()
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, p_s)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, num_p=1):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_p=num_p)

    return model

