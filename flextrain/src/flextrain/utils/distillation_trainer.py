"""Trainer implementation for optional teacher-student distillation."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union, cast

import torch
from accelerate.utils import DistributedType

from flexrank.utils import init_logger
from flextrain.utils.args import DistillationTrainingArguments

from .custom_trainer import CustomTrainer as Trainer

log = init_logger(__name__)

__all__ = ["DistillationTrainer"]


def xor(a: bool, b: bool) -> bool:
    """Return exclusive-or for two booleans."""
    return a != b


class DistillationTrainer(Trainer):
    """Trainer that mixes supervised loss with optional teacher KL loss."""

    def __init__(self, *p_args, teacher_model: torch.nn.Module = None, **kwargs):
        super().__init__(*p_args, **kwargs)
        args = self.args
        assert isinstance(args, DistillationTrainingArguments), (
            f"DistillationTrainer must receive DistillationTrainingArguments, got {type(args)}"
        )
        assert not xor(args.kl_loss_w > 0, bool(teacher_model)), (
            f"Invalid combination of inputs: {args.kl_loss_w=} and "
            f"teacher_model is {bool(teacher_model)}"
        )

        if teacher_model is not None:
            teacher_model.requires_grad_(False).eval()
            if self.accelerator.distributed_type == DistributedType.FSDP:
                # Keep the known FSDP workaround: some HF/Accelerate stacks only wrap correctly
                # when the model goes through `prepare` together with an optimizer.
                dummy_opt = torch.optim.SGD(teacher_model.parameters(), lr=0)
                teacher_model, _ = self.accelerator.prepare(teacher_model, dummy_opt)
            elif self.args.torch_compile:
                # The teacher is inference-only and does not participate in gradient
                # synchronization. Keeping it as a local eager module avoids adding
                # extra DDP/compile hook state around the teacher during compiled
                # distributed training.
                teacher_model = teacher_model.to(self.accelerator.device)
                teacher_model = self._compile_teacher(teacher_model)
            else:
                teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher = teacher_model
        self._teacher_device = self._get_module_device(self.teacher)
        self._teacher_stream: Optional[torch.cuda.Stream] = None
        self._teacher_stream_device: Optional[torch.device] = None

    @property
    def uses_distillation(self) -> bool:
        """Return whether a teacher model is configured."""
        return bool(self.teacher)

    def _compile_teacher(self, teacher_model: torch.nn.Module) -> torch.nn.Module:
        """Compile the local inference-only teacher with Trainer compile settings."""
        compile_kwargs = {}
        if self.args.torch_compile_backend is not None:
            compile_kwargs["backend"] = self.args.torch_compile_backend
        if self.args.torch_compile_mode is not None:
            compile_kwargs["mode"] = self.args.torch_compile_mode

        compile_inplace = getattr(teacher_model, "compile", None)
        if callable(compile_inplace):
            compile_inplace(**compile_kwargs)
            return teacher_model
        return torch.compile(teacher_model, **compile_kwargs)

    @staticmethod
    def _get_module_device(module: Optional[torch.nn.Module]) -> Optional[torch.device]:
        if module is None:
            return None
        try:
            return next(module.parameters()).device
        except StopIteration:
            return None

    @staticmethod
    def _supports_use_cache(module: Optional[torch.nn.Module]) -> bool:
        """Return whether a possibly wrapped HF model exposes a use_cache config."""
        while module is not None:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                return True
            module = getattr(module, "module", None) or getattr(module, "_orig_mod", None)
        return False

    def _move_to_device(self, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device == device:
                return value
            return value.to(device=device, non_blocking=True)
        if isinstance(value, Mapping):
            return {k: self._move_to_device(v, device) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(v, device) for v in value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._move_to_device(v, device) for v in value]
        return value

    def _prepare_teacher_inputs(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        teacher_inputs = {
            k: v for k, v in inputs.items() if k not in {"labels", "num_items_in_batch"}
        }
        if self._supports_use_cache(self.teacher):
            teacher_inputs.setdefault("use_cache", False)
        if self._teacher_device is None:
            return teacher_inputs
        return self._move_to_device(teacher_inputs, self._teacher_device)

    def _can_overlap_teacher_forward(self, student_device: Optional[torch.device]) -> bool:
        teacher_device = self._teacher_device
        if student_device is None or teacher_device is None:
            return False
        if student_device.type != "cuda" or teacher_device.type != "cuda":
            return False
        if student_device != teacher_device:
            return False
        if self.args.torch_compile:
            # `torch.compile` is much more sensitive to side-stream execution than
            # eager mode. Keep teacher/student forwards serialized when compile is
            # enabled to avoid stream ordering hazards during distributed training.
            return False
        # FSDP forwards may issue collectives; keep them serialized to avoid overlap hazards.
        return self.accelerator.distributed_type != DistributedType.FSDP

    def _get_teacher_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._teacher_stream is None or self._teacher_stream_device != device:
            self._teacher_stream = torch.cuda.Stream(device=device)
            self._teacher_stream_device = device
        return self._teacher_stream

    def _forward_teacher(self, teacher_inputs: dict[str, Union[torch.Tensor, Any]]):
        with torch.inference_mode():
            return self.teacher(**teacher_inputs)

    def _compute_kl_loss(
        self,
        inputs,
        out_s,
        out_t,
        temperature: float,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert out_s.logits.size() == out_t.logits.size()

        assert "attention_mask" in inputs, "Missing attention_mask in model inputs"
        token_mask = inputs["attention_mask"].to(out_s.logits.device).reshape(-1).bool()

        vocab_size = out_s.logits.size(-1)
        out_s_m_logits = out_s.logits.reshape(-1, vocab_size)[token_mask]
        out_t_m_logits = out_t.logits.to(out_s.logits.device).reshape(-1, vocab_size)[token_mask]

        assert out_s_m_logits.size() == out_t_m_logits.size()

        kl_loss_tot = torch.nn.functional.kl_div(
            input=torch.nn.functional.log_softmax(out_s_m_logits / temperature, dim=-1),
            target=torch.nn.functional.softmax(out_t_m_logits / temperature, dim=-1),
            reduction="sum",
        ) * (temperature**2)

        # Divide by number of active tokens in the batch (already GA-aware in Trainer)
        denom = (
            num_items_in_batch
            if num_items_in_batch is not None
            else inputs.get("num_items_in_batch")
        )
        assert denom is not None, (
            "num_items_in_batch is required for KL normalization during training"
        )
        # Ensure same device/dtype (and avoid divide-by-zero edge cases)
        denom = denom.to(device=kl_loss_tot.device, dtype=kl_loss_tot.dtype).clamp_min(1)
        return kl_loss_tot / denom

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        if self._supports_use_cache(model):
            inputs.setdefault("use_cache", False)

        args = cast(DistillationTrainingArguments, self.args)
        teacher_inputs = None
        teacher_outputs = None
        teacher_stream = None

        if model.training and args.kl_loss_w > 0:
            teacher_inputs = self._prepare_teacher_inputs(inputs)
            student_device = self._get_module_device(model)
            if self._can_overlap_teacher_forward(student_device):
                teacher_stream = self._get_teacher_stream(student_device)
                with torch.cuda.stream(teacher_stream):
                    teacher_outputs = self._forward_teacher(teacher_inputs)

        # Obtain the output from the model
        out_s = model(**inputs)

        if not model.training:
            loss = out_s.loss
        else:
            loss = out_s.loss.new_zeros(())  # tensor scalar on correct device/dtype

            if num_items_in_batch is not None:
                if args.ce_loss_w > 0:
                    # Robustness for HF label smoothing
                    if getattr(self, "label_smoother", None) is not None and "labels" in inputs:
                        ce_loss = self.label_smoother(out_s, inputs["labels"])
                    else:
                        ce_loss = out_s.loss
                    loss += args.ce_loss_w * ce_loss

                if args.kl_loss_w > 0:
                    if teacher_outputs is None:
                        teacher_outputs = self._forward_teacher(teacher_inputs)
                    elif teacher_stream is not None:
                        torch.cuda.current_stream(device=out_s.logits.device).wait_stream(
                            teacher_stream
                        )
                    kl_loss = self._compute_kl_loss(
                        inputs,
                        out_s,
                        teacher_outputs,
                        args.temperature,
                        num_items_in_batch,
                    )
                    loss += args.kl_loss_w * kl_loss
            else:
                # Fallback path (should be rare in training): just use model loss
                loss = out_s.loss

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, out_s) if return_outputs else loss

    def _has_sampler_callback(self) -> bool:
        return any(
            type(cb).__name__ == "SamplerCallback"
            and type(cb).__module__ == "flextrain.callbacks.sampler_callback"
            for cb in self.callback_handler.callbacks
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if not self._has_sampler_callback():
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # The sampler callback owns submodel evaluation.
        metrics = {}
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics
