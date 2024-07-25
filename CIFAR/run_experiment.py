import os
import re
import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional, List, Tuple

import datasets
import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "{epoch:02d}-{step:02d}"


class CIFAR10Module(models.ASTBaseModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: dict,
        model_name: str,
        model_params: dict,
        optimizer_name: str,
        optimizer_config: dict,
        loss_name: str = "SelfModelingLoss",
        loss_params: dict = {"aux_task_weight": 1.0},
        num_classes: int = 10,
        self_modeling_target_layers: List[str] = [],
        random_seed: Optional[int] = None,
    ):
        """
        A specialized module for CIFAR-10 dataset training using models from the `models` module with self-modeling capabilities.

        Args:
        dataset_name (str): Name of the dataset class from the datasets module.
        dataset_config (dict): Configuration parameters for the dataset module.
        model_name (str): Name of the model class from the models module.
        model_params (dict): Parameters to initialize the model class.
        optimizer_name (str): Name of the optimizer to use.
        optimizer_config (dict): Configuration for the optimizer.
        num_classes (int): Number of output classes.
        aux_task_weight (float): Weighting factor for auxiliary task loss in model training.
        self_modeling_target_layers (List[str]): Specific layers of the model used for self-modeling loss calculations.
        random_seed (Optional[int]): Seed for random operations to ensure reproducibility.

        Initializes the module, sets up the random seed, and creates the model and data module based on the provided specifications.
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.model_params = model_params
        self.num_classes = num_classes
        self.self_modeling_target_layers = self_modeling_target_layers
        self.random_seed = random_seed

        models.configure_random_seed(random_seed)
        model = self.create_model()
        model_architecture = models.get_model_architecture(model)

        super().__init__(
            model_name=model_name,
            model_params=model_params,
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config,
            loss_name=loss_name,
            loss_params=loss_params,
            num_classes=num_classes,
            self_modeling_target_layers=self_modeling_target_layers,
            model_architecture=model_architecture,
        )
        self.model = model
        self.register_target_layer_activations_hooks()

    def create_model(self) -> nn.Module:
        """
        Creates and returns a model instance from the model parameters provided at initialization.

        Returns:
        nn.Module: The instantiated model with a potentially modified output dimension to accommodate self-modeling layers.
        """
        model_class = getattr(models, self.model_name)
        model = model_class(output_dim=self.num_classes, **self.model_params)
        dummy_input = torch.randn((1, 3, 32, 32))  # Example input tensor for the model.
        total_activation_size = models.calculate_total_activation_size(
            model, dummy_input, self.self_modeling_target_layers
        )
        model = model_class(
            output_dim=self.num_classes + total_activation_size, **self.model_params
        )
        return model

    def prepare_data(self) -> None:
        """
        Prepares the data for training by instantiating the dataset class and loading the data.
        """
        dataset_class = getattr(datasets, self.dataset_name)
        self.data_module = dataset_class(
            random_seed=self.random_seed, **self.dataset_config
        )
        super().prepare_data()


def run_experiment(
    module: CIFAR10Module,
    max_epochs: int,
    self_modeling_target_layers: List[str],
    logger_suffix: str = "",
    pretrained_version: Optional[str] = None,
) -> str:
    """
    Runs training and testing experiments on CIFAR-10 data using the specified module setup.

    Args:
    module (CIFAR10Module): The training module setup.
    max_epochs (int): Maximum number of epochs for training.
    self_modeling_target_layers (List[str]): Layers targeted for self-modeling in the model.
    logger_suffix (str, optional): Suffix to append to the logger name for differentiation.
    pretrained_version (Optional[str]): Version identifier of a pretrained model to continue training.
    skip_rlct (bool): Whether to skip the RLCT computation after training.

    Returns:
    str: The version identifier for the TensorBoard logs associated with the run.
    """
    logger_name = f"CIFAR-10_{module.model_name}_{logger_suffix if logger_suffix else '_'.join(self_modeling_target_layers)}"
    tb_logger = TensorBoardLogger(
        "./artifacts/logs/", name=logger_name, version=pretrained_version
    )
    checkpoint_callback = ModelCheckpoint(
        filename=CHECKPOINT_FILE, every_n_epochs=10, save_top_k=-1
    )

    # Setup Trainer
    trainer_args = {
        "max_epochs": max_epochs,
        "accelerator": DEVICE,
        "logger": tb_logger,
        "callbacks": [checkpoint_callback],
    }

    last_checkpoint, last_epoch = None, 0
    if pretrained_version is not None:
        all_checkpoints = _get_checkpoints_info(tb_logger.log_dir)
        if all_checkpoints:
            last_checkpoint, _, last_epoch = all_checkpoints[-1]
            last_epoch += 1  # Start next epoch

    if (pretrained_version is None) or (max_epochs > last_epoch):
        print(f"Training from epoch {last_epoch} to {max_epochs}...")
        trainer = L.Trainer(**trainer_args)
        trainer.fit(module, ckpt_path=last_checkpoint)
        trainer.test(module)

    return tb_logger.version


def _get_checkpoints_info(log_dir: str) -> List[Tuple[str, int, int]]:
    """
    Retrieves a list of checkpoint files along with their corresponding training steps and epochs.

    Args:
    log_dir (str): Directory where the log files and checkpoints are stored.

    Returns:
    List[Tuple[str, int, int]]: A list of tuples containing the checkpoint file path, step number, and epoch number.
    """
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".ckpt")
    ]
    checkpoint_files_with_step = []
    for checkpoint in checkpoint_files:
        step = int(re.search(r"step=(\d+)", checkpoint).group(1))
        epoch = int(re.search(r"epoch=(\d+)", checkpoint).group(1))
        checkpoint_files_with_step.append((checkpoint, step, epoch))

    checkpoint_files_with_step.sort(key=lambda x: x[1])  # Sort by step number
    return checkpoint_files_with_step


# Available layers: ['conv1(Conv2d)', 'bn1(BatchNorm2d)', 'activation_bn1(ReLU)', 'layer1(Sequential)',
# 'layer1.0(ResNetBlock)', 'layer1.0.conv1(Conv2d)', 'layer1.0.bn1(BatchNorm2d)', 'layer1.0.activation_bn1(ReLU)',
# 'layer1.0.conv2(Conv2d)', 'layer1.0.bn2(BatchNorm2d)', 'layer1.0.shortcut(Sequential)',
# 'layer1.0.activation_shortcut(ReLU)', 'layer1.1(ResNetBlock)', 'layer1.1.conv1(Conv2d)', 'layer1.1.bn1(BatchNorm2d)',
# 'layer1.1.activation_bn1(ReLU)', 'layer1.1.conv2(Conv2d)', 'layer1.1.bn2(BatchNorm2d)', 'layer1.1.shortcut(Sequential)',
# 'layer1.1.activation_shortcut(ReLU)', 'layer2(Sequential)', 'layer2.0(ResNetBlock)', 'layer2.0.conv1(Conv2d)',
# 'layer2.0.bn1(BatchNorm2d)', 'layer2.0.activation_bn1(ReLU)', 'layer2.0.conv2(Conv2d)', 'layer2.0.bn2(BatchNorm2d)',
# 'layer2.0.shortcut(Sequential)', 'layer2.0.shortcut.0(Conv2d)', 'layer2.0.shortcut.1(BatchNorm2d)',
# 'layer2.0.activation_shortcut(ReLU)', 'layer2.1(ResNetBlock)', 'layer2.1.conv1(Conv2d)', 'layer2.1.bn1(BatchNorm2d)',
# 'layer2.1.activation_bn1(ReLU)', 'layer2.1.conv2(Conv2d)', 'layer2.1.bn2(BatchNorm2d)', 'layer2.1.shortcut(Sequential)',
# 'layer2.1.activation_shortcut(ReLU)', 'layer3(Sequential)', 'layer3.0(ResNetBlock)', 'layer3.0.conv1(Conv2d)',
# 'layer3.0.bn1(BatchNorm2d)', 'layer3.0.activation_bn1(ReLU)', 'layer3.0.conv2(Conv2d)', 'layer3.0.bn2(BatchNorm2d)',
# 'layer3.0.shortcut(Sequential)', 'layer3.0.shortcut.0(Conv2d)', 'layer3.0.shortcut.1(BatchNorm2d)',
# 'layer3.0.activation_shortcut(ReLU)', 'layer3.1(ResNetBlock)', 'layer3.1.conv1(Conv2d)', 'layer3.1.bn1(BatchNorm2d)',
# 'layer3.1.activation_bn1(ReLU)', 'layer3.1.conv2(Conv2d)', 'layer3.1.bn2(BatchNorm2d)', 'layer3.1.shortcut(Sequential)',
# 'layer3.1.activation_shortcut(ReLU)', 'layer4(Sequential)', 'layer4.0(ResNetBlock)', 'layer4.0.conv1(Conv2d)',
# 'layer4.0.bn1(BatchNorm2d)', 'layer4.0.activation_bn1(ReLU)', 'layer4.0.conv2(Conv2d)', 'layer4.0.bn2(BatchNorm2d)',
# 'layer4.0.shortcut(Sequential)', 'layer4.0.shortcut.0(Conv2d)', 'layer4.0.shortcut.1(BatchNorm2d)',
# 'layer4.0.activation_shortcut(ReLU)', 'layer4.1(ResNetBlock)', 'layer4.1.conv1(Conv2d)', 'layer4.1.bn1(BatchNorm2d)',
# 'layer4.1.activation_bn1(ReLU)', 'layer4.1.conv2(Conv2d)', 'layer4.1.bn2(BatchNorm2d)', 'layer4.1.shortcut(Sequential)',
# 'layer4.1.activation_shortcut(ReLU)',  'hidden_linear'(Linear)', 'activation_fc(ReLU)', 'linear(Linear)']


if __name__ == "__main__":
    for random_seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        variants = {
            "baseline": CIFAR10Module(
                dataset_name="CIFAR10DataModule",
                dataset_config={
                    "batch_size": 64,
                    "data_augmentation": True,
                },
                model_name="ResNet18WithHiddenLayer",
                model_params={},
                optimizer_name="Adam",
                optimizer_config={"lr": 1e-4},
                loss_name="SelfModelingLoss",
                loss_params={"aux_task_weight": 1.0},
                num_classes=10,
                self_modeling_target_layers=[
                ],
                random_seed=random_seed,
            ),
            "fourth_block_and_hidden_layer_AW=1.0": CIFAR10Module(
                dataset_name="CIFAR10DataModule",
                dataset_config={
                    "batch_size": 64,
                    "data_augmentation": True,
                },
                model_name="ResNet18WithHiddenLayer",
                model_params={},
                optimizer_name="Adam",
                optimizer_config={"lr": 1e-4},
                loss_name="SelfModelingLoss",
                loss_params={"aux_task_weight": 1.0},
                num_classes=10,
                self_modeling_target_layers=[
                    "layer4.1.activation_shortcut",
                    "activation_fc",
                ],
                random_seed=random_seed,
            ),
            "fourth_block_and_hidden_layer_AW=0.5": CIFAR10Module(
                dataset_name="CIFAR10DataModule",
                dataset_config={
                    "batch_size": 64,
                    "data_augmentation": True,
                },
                model_name="ResNet18WithHiddenLayer",
                model_params={},
                optimizer_name="Adam",
                optimizer_config={"lr": 1e-4},
                loss_name="SelfModelingLoss",
                loss_params={"aux_task_weight": 0.5},
                num_classes=10,
                self_modeling_target_layers=[
                    "layer4.1.activation_shortcut",
                    "activation_fc",
                ],
                random_seed=random_seed,
            ),
            "fourth_block_and_hidden_layer_AW=2": CIFAR10Module(
                dataset_name="CIFAR10DataModule",
                dataset_config={
                    "batch_size": 64,
                    "data_augmentation": True,
                },
                model_name="ResNet18WithHiddenLayer",
                model_params={},
                optimizer_name="Adam",
                optimizer_config={"lr": 1e-4},
                loss_name="SelfModelingLoss",
                loss_params={"aux_task_weight": 2},
                num_classes=10,
                self_modeling_target_layers=[
                    "layer4.1.activation_shortcut",
                    "activation_fc",
                ],
                random_seed=random_seed,
            ),
        }
        for variant_name, module in variants.items():
            print(f"Running experiment for variant: {variant_name}, seed: {random_seed}...")
            run_experiment(
                module,
                max_epochs=250,
                self_modeling_target_layers=module.self_modeling_target_layers,
                logger_suffix=variant_name,
            )
