"""Training utilities used by FlexRank"""

from torch.nn import Module

from .logger import init_logger

logger = init_logger(__name__)


def train_only_specific_layers(
    network: Module,
    specific_layers: list[tuple[str, Module]],
    trainable_bias: bool = False,
):
    """Set as trainable only the layers that are supposed to be trained"""

    frozen, unfrozen = [], []

    for name, param in network.named_parameters():
        if not trainable_bias and "bias" in name:
            frozen.append(name)
            param.requires_grad = False
            continue

        if not any(layer_name in name for layer_name, _ in specific_layers):
            frozen.append(name)
            param.requires_grad = False  # Freeze layer
        else:
            unfrozen.append(name)
            param.requires_grad = True  # Unfreeze selected layer
    newline = "\n"
    logger.info("Frozen layers:\n%s", newline.join(frozen))
    logger.info("Unfrozen layers:\n%s", newline.join(unfrozen))
