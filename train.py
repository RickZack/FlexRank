"""Entry point for FlexRank knowledge consolidation training."""

from flextrain.dataset import load_dataset_from_hf
from flextrain.model import load_model_from_hf
from flextrain.utils import init_logger
from flextrain.utils.args import Config, parse_structured_config
from flextrain.utils.flexrank_utils import get_flexrank
from flextrain.utils.wandb_utils import setup_wandb

log = init_logger(__name__)


@parse_structured_config(version_base=None, config_path="config", config_name="config")
def main(args: Config) -> None:
    """Main function for FlexRank training."""
    setup_wandb(
        wandb_args=args.logger,
        run_config=args,
        output_dir=args.train.output_dir,
        load_path=args.load_flexrank_path,
    )

    # Model
    model_data = load_model_from_hf(args.task, args.model)

    # Datasets
    ds_splits = load_dataset_from_hf(
        args.task,
        args.dataset,
        model_data.proc,
        args.train.data_seed,
    )

    # Actual training
    flexrank_data = get_flexrank(args, model_data, ds_splits)
    log.info("Starting Knowledge Consolidation training...")
    flexrank_data.trainer.train(resume_from_checkpoint=args.resume_checkpoint_path)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
