"""DINOv3 image-classification wrapper with a linear classifier head."""

from typing import Optional
import os
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    PreTrainedModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
)
from transformers.modeling_outputs import ImageClassifierOutput


class DINOv3ViTForImageClassification(PreTrainedModel):
    """
    DINOv3ViT backbone + linear classification head.

    Supports two use cases:

    1) Start from a backbone checkpoint:
       model = AutoModelForImageClassification.from_pretrained(
           "facebook/dinov3-vit-base",  # example
           num_labels=...,              # required
           classifier_dropout=...,      # optional
           id2label=..., label2id=...,  # optional
       )

       => loads backbone weights, randomly initialized head.

    2) Reload a classifier checkpoint saved with save_pretrained():
       model = AutoModelForImageClassification.from_pretrained("path/to/checkpoint")
    """

    # IMPORTANT: use the backbone config class here
    config_class = DINOv3ViTConfig
    main_input_name = "pixel_values"

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__(config)

        # Make sure we have num_labels
        assert hasattr(config, "num_labels"), "DINOv3ViTForImageClassification needs num_labels"
        self.num_labels = config.num_labels

        # Backbone
        self.backbone = DINOv3ViTModel(config)

        # Hidden size
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "embed_dim"):
            hidden_size = config.embed_dim
        else:
            raise ValueError(
                "Cannot infer hidden size from config (expected hidden_size or embed_dim)."
            )

        # Head
        dropout_prob = getattr(config, "classifier_dropout", None) or 0.0
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        # Match the classifier-head initialization used by DINOv3 linear eval.
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> ImageClassifierOutput:
        """Run image classification and optionally compute cross-entropy loss."""
        outputs = self.backbone(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0]  # [B, 1, D]
        last = self.dropout(cls_token)
        logits = self.classifier(last)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[DINOv3ViTConfig] = None,
        **kwargs,
    ):
        if config is None:
            # Pass kwargs here so things like num_labels are respected during config load
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 1. Determine if this is a checkpoint that ALREADY HAS a classifier.
        # Check architectures list OR check if 'num_labels' was already saved in the config.
        # Usually, backbone configs don't have 'num_labels' until you fine-tune them.
        architectures = getattr(config, "architectures", [])
        is_full_classifier = (
            cls.__name__ in architectures
            or os.path.isdir(pretrained_model_name_or_path)
            or "classifier.weight" in kwargs.get("state_dict", {})  # Manual check if provided
        )

        # CASE 1: Load the full saved model (weights + config)
        if is_full_classifier:
            return super(DINOv3ViTForImageClassification, cls).from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs,
            )

        # CASE 2: Initialize new head on top of a backbone
        num_labels = kwargs.pop("num_labels", getattr(config, "num_labels", 1000))
        classifier_dropout = kwargs.pop(
            "classifier_dropout", getattr(config, "classifier_dropout", None)
        )

        config.num_labels = num_labels
        if classifier_dropout is not None:
            config.classifier_dropout = classifier_dropout

        model = cls(config)

        # Load backbone only
        backbone_kwargs = kwargs.copy()
        backbone_kwargs.pop("config", None)
        backbone_kwargs.pop("ignore_mismatched_sizes", None)

        backbone = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path,
            **backbone_kwargs,
        )
        model.backbone.load_state_dict(backbone.state_dict())

        return model


AutoModelForImageClassification.register(
    DINOv3ViTConfig,
    DINOv3ViTForImageClassification,
)
