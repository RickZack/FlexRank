"""Checkpoint callbacks for Hugging Face Trainer workflows."""

from transformers import TrainerCallback, TrainerState, TrainerControl

__all__ = ["SaveAtBeginEndCallback"]


class SaveAtBeginEndCallback(TrainerCallback):
    """Force checkpoint saves at the first and final training steps."""

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Mark the first step for checkpointing."""
        is_first_step = not state.global_step
        control.should_save = is_first_step or control.should_save
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Mark the final step for checkpointing."""
        # Save at the very end of training
        if state.max_steps is not None and state.max_steps > 0:
            is_last_step = state.global_step == state.max_steps
            control.should_save = is_last_step or control.should_save
        return control
