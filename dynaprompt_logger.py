import json
import os
import time
from collections import defaultdict

import torch
import torchvision.utils as vutils


import torch
import json
from pathlib import Path


class DynaPromptExperimentLogger:

    def __init__(self, root_dir="dynaprompt_runs"):
        """Initialize the experiment logger.

        Args:
            root_dir (str): Directory where per-sample run folders and JSON
                metadata files will be written. The directory is created if it
                does not exist.

        Attributes created:
            root_dir (Path): Path object for the root logging directory.
            predictions (list): Accumulates prediction dictionaries recorded
                during an experiment run; kept in-memory for later use.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []


    def _save_tensor(self, tensor, path):
        """Save a PyTorch tensor to disk.

        The tensor is detached from the computation graph and moved to CPU
        before being serialized with `torch.save`.

        Args:
            tensor (torch.Tensor): Tensor to save.
            path (str or Path): Filesystem path where tensor will be saved.
        """
        torch.save(tensor.detach().cpu(), path)

    def save_image(self, tensor, path):
        """Save an image tensor to disk as a normalized image file.

        This normalizes the tensor into the [0,1] range per-tensor and uses
        `torchvision.utils.save_image` to write the image. The tensor is
        detached and moved to CPU first.

        Args:
            tensor (torch.Tensor): Image tensor of shape (C,H,W) or (H,W,C).
            path (str or Path): Path to save the image file.
        """
        img = tensor.detach().cpu()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        vutils.save_image(img, path)

    # ---------- SAMPLE START ----------
    def start_sample(self, image_tensor, label, image_name):
        """Begin logging for a single sample.

        Creates a per-sample folder and records basic metadata (image name,
        label and timestamp). Also saves the provided original image to
        disk as `original.png` inside the sample folder.

        Args:
            image_tensor (torch.Tensor): Original image tensor (C,H,W).
            label (int|str|torch.Tensor): Ground-truth label for the sample.
            image_name (str): Unique identifier / filename for this sample.
        """
        self.image_name = image_name
        self.sample_dir = self.root_dir / image_name
        self.sample_dir.mkdir(exist_ok=True)

        self.log_data = {
            "image_name": image_name,
            "label": int(label) if isinstance(label, torch.Tensor) else label,
            "timestamp": time.time(),
        }

        # save original image
        self.save_image(image_tensor, self.sample_dir / "original.png")

    # def log_views_and_features(self, views, image_features):
    #     views_dir = self.sample_dir / "views"
    #     views_dir.mkdir(exist_ok=True)

    #     for i, img in enumerate(views):
    #         self.save_image(img, views_dir / f"view_{i}.png")

    #     tensor_dir = self.sample_dir / "tensors"
    #     tensor_dir.mkdir(exist_ok=True)

    #     self._save_tensor(image_features, tensor_dir / "image_features.pt")

        # ---------- AUGMENTATIONS ----------
    def log_augmentations(self, aug_tensor):
        """Save a batch of augmented views for the current sample.

        Writes each augmentation to the sample's `augmentations/` folder and
        records the number of views in the sample metadata. Expected input
        shape is (N, C, H, W).

        Args:
            aug_tensor (torch.Tensor): Tensor containing augmented views.
        """
        aug_dir = self.sample_dir / "augmentations"
        aug_dir.mkdir(exist_ok=True)

        self.log_data["num_augmentations"] = aug_tensor.shape[0]

        for i in range(aug_tensor.shape[0]):
            self.save_image(aug_tensor[i], aug_dir / f"{i}.png")

    # def log_text_features(self, text_features):
    #     tensor_dir = self.sample_dir / "tensors"
    #     self._save_tensor(text_features, tensor_dir / "text_features.pt")

    # def log_logits_probs(self, logits, probs):
    #     tensor_dir = self.sample_dir / "tensors"
    #     self._save_tensor(logits, tensor_dir / "logits.pt")
    #     self._save_tensor(probs, tensor_dir / "probs.pt")

    # def log_entropy_metrics(self, entropy_view, entropy_prompt):
    #     tensor_dir = self.sample_dir / "tensors"
    #     self._save_tensor(entropy_view, tensor_dir / "entropy_view.pt")
    #     self._save_tensor(entropy_prompt, tensor_dir / "entropy_prompt.pt")

    # def log_prompt_texts(self, prompt_buffer, selected_ids, updated_ids):
    #     def get_text(p):
    #         if hasattr(p, "text"):
    #             return p.text
    #         if hasattr(p, "token_ids"):
    #             return self.tokenizer.decode(p.token_ids)
    #         return str(p)

    #     metadata = {
    #         "prompt_buffer": {i: get_text(p) for i, p in enumerate(prompt_buffer)},
    #         "selected_prompts": {i: get_text(prompt_buffer[i]) for i in selected_ids},
    #         "updated_prompts": {i: get_text(prompt_buffer[i]) for i in updated_ids},
    #     }

        # with open(self.sample_dir / "metadata.json", "w") as f:
        #     json.dump(metadata, f, indent=2)

    def log_confidence(self, confidences):
        """Record confidence scores for each augmentation.

        Stores a mapping of augmentation index -> confidence in `log_data` and
        also records the maximum confidence as `final_confidence` for use in
        naming or thresholding decisions later.

        Args:
            confidences (Sequence|torch.Tensor): Per-augmentation
                confidence scores.
        """
        conf_dict = {int(i): float(confidences[i]) for i in range(len(confidences))}
        self.log_data["confidence_per_aug"] = conf_dict

        self.final_confidence = max(conf_dict.values())

    # ---------- ENTROPY CRITERIA ----------
    def log_entropy_selection(self, metrics):
        """Record entropy metrics used to select prompts.

        Args:
            metrics (dict): Mapping from prompt id to entropy and selection
                metadata. Stored verbatim into `log_data` under
                `entropy_selection`.
        """
        self.log_data["entropy_selection"] = metrics

    def log_entropy_all_views(self, entropy_view):
        """
        Docstring for log_entropy_all_views
        
        :param self: Description
        :param entropy_view: Description
        """        
        self.log_data["entropy_all_views"] = entropy_view


    # ---------- PROBABILITY DIFFERENCE ----------
    def log_prob_diff_selection(self, metrics):
        """Record probability-difference (PLPD) metrics used for selection.

        Args:
            metrics (dict): Mapping from prompt id to probability-difference
                statistics (plpd) and selection flag. Stored into
                `log_data` under `probability_difference_selection`.
        """
        self.log_data["probability_difference_selection"] = metrics

    def log_threshold(self, threshold):
        """Record the confidence threshold used for filtering samples.

        Args:
            threshold (float): Selection percentile or threshold value.
        """
        self.log_data["confidence_threshold"] = float(threshold)

    def log_filtered_count(self, count):
        """Record how many augmentations/samples passed the selection stage.

        Args:
            count (int): Number of selected items.
        """
        self.log_data["num_selected_for_prompt_selection"] = int(count)

    # ---------- PROMPT BUFFER ----------
    def log_prompt_buffer(self, buffer_prompts):
        """Store the current ordering or contents of the prompt buffer.

        Args:
            buffer_prompts (list): Representation of prompt buffer (ids,
                counts, or other diagnostics). Stored for debugging.
        """
        self.log_data["prompt_buffer"] = buffer_prompts

    def log_updated_prompts(self, updated_prompts):
        """Record updated prompt order or stats after selection/usage.

        Args:
            updated_prompts (list): Updated ordering or metadata for prompts.
        """
        self.log_data["updated_prompts"] = updated_prompts

    # ---------- FINAL PROMPTS ----------
    def log_final_selected_prompts(self, prompts):
        """Save the final list of prompts selected for adaptation.

        Args:
            prompts (list|int): Final selected prompt identifiers or textual
                representations. Stored under `final_selected_prompts`.
        """
        self.log_data["final_selected_prompts"] = prompts

    def log_prediction(
        self,
        raw_pred,
        adapted_pred,
        raw_entropy,
        adapted_entropy,
        raw_conf,
        adapted_conf,
    ):
        """Append prediction and metric information for a single sample.

        This keeps an in-memory list of dictionaries containing both the raw
        (pre-adaptation) and adapted (post-adaptation) predictions along with
        associated entropy and confidence metrics. The list can be used for
        later analysis or writing to disk.

        Args:
            raw_pred: The model's raw/unadapted logits or predicted id.
            adapted_pred: The model's logits or predicted id after adaptation.
            raw_entropy (float): Entropy value before adaptation.
            adapted_entropy (float): Entropy value after adaptation.
            raw_conf (float): Confidence before adaptation.
            adapted_conf (float): Confidence after adaptation.
        """
        self.predictions.append({
            "raw_pred": raw_pred,
            "adapted_pred": adapted_pred,
            "raw_entropy": raw_entropy,
            "adapted_entropy": adapted_entropy,
            "raw_conf": raw_conf,
            "adapted_conf": adapted_conf,
        })

        # ---------- SAVE JSON ----------

    def end_sample(self):
        """Finalize logging for the current sample and write metadata JSON.

        Uses `final_confidence` to create a unique filename and writes the
        accumulated `log_data` to disk as a JSON file in the sample folder.
        """
        final_conf = getattr(self, "final_confidence", 0.0)
        json_name = f"{self.image_name}_{final_conf:.4f}.json"
        json_path = self.sample_dir / json_name

        with open(json_path, "w") as f:
            json.dump(self.log_data, f, indent=2)
