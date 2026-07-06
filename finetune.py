"""Entry point for supervised finetuning of selected model layers."""

from torch.optim import SGD
from transformers import Trainer

from flextrain.utils.wandb_utils import setup_wandb
from flextrain.dataset import load_dataset_from_hf
from flextrain.model import load_model_from_hf, set_trainable_layers
from flextrain.utils import init_logger
from flextrain.utils.finetune_args import FinetuneConfig, parse_structured_config

log = init_logger(__name__)


@parse_structured_config(version_base=None, config_path="config", config_name="finetune")
def main(args: FinetuneConfig) -> None:
    """Main function for finetuning."""
    setup_wandb(
        wandb_args=args.logger,
        run_config=args,
        output_dir=args.train.output_dir,
    )

    # Model
    model_data = load_model_from_hf(args.task, args.model)
    set_trainable_layers(model_data.model, args.layers_to_train)

    # Datasets
    ds_splits = load_dataset_from_hf(
        args.task,
        args.dataset,
        model_data.proc,
        args.train.data_seed,
    )

    # Actual training
    trainer = Trainer(
        model=model_data.model,
        args=args.hf_train,
        train_dataset=ds_splits.train,
        eval_dataset=ds_splits.val,
        data_collator=ds_splits.collator,
        processing_class=model_data.proc,
        compute_metrics=model_data.metric_fn,
        optimizer_cls_and_kwargs=(
            SGD,
            {
                "lr": args.train.learning_rate,
                "weight_decay": args.train.weight_decay,
                "momentum": args.train.adam_beta1,
            },
        ),
    )

    log.info("Starting finetuning...")
    trainer.train(resume_from_checkpoint=args.resume_checkpoint_path)
    log.info(trainer.evaluate())
    trainer.save_model()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
