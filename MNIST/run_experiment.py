import os
import re
import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional, List, Tuple
from tensorboard.backend.event_processing import event_accumulator

import datasets
import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "{epoch:02d}-{step:02d}"


class MNISTModule(models.ASTBaseModule):
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
        A specialized module for MNIST dataset training using models from the `models` module with self-modeling capabilities.

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
        dummy_input = torch.randn((1, 1, 28, 28))  # Example input tensor for the model.
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
    module: MNISTModule,
    max_epochs: int,
    self_modeling_target_layers: List[str],
    logger_suffix: str = "",
    pretrained_version: Optional[str] = None,
    skip_rlct: bool = False,
) -> str:
    """
    Runs training and testing experiments on MNIST data using the specified module setup.

    Args:
    module (MNISTModule): The training module setup.
    max_epochs (int): Maximum number of epochs for training.
    self_modeling_target_layers (List[str]): Layers targeted for self-modeling in the model.
    logger_suffix (str, optional): Suffix to append to the logger name for differentiation.
    pretrained_version (Optional[str]): Version identifier of a pretrained model to continue training.
    skip_rlct (bool): Whether to skip the RLCT computation after training.

    Returns:
    str: The version identifier for the TensorBoard logs associated with the run.
    """
    logger_name = f"MNIST_{logger_suffix if logger_suffix else '_'.join(self_modeling_target_layers)}"
    tb_logger = TensorBoardLogger(
        "./artifacts/logs/", name=logger_name, version=pretrained_version
    )
    checkpoint_callback = ModelCheckpoint(
        filename=CHECKPOINT_FILE, every_n_epochs=1, save_top_k=-1
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
        if last_epoch == 0:
            trainer.validate(module)
            trainer.save_checkpoint(os.path.join(tb_logger.log_dir, "checkpoints", "epoch=00-step=00.ckpt"))
        trainer.fit(module, ckpt_path=last_checkpoint)
        trainer.test(module)

    if not skip_rlct:
        module.prepare_data()
        train_loader = module.data_module.train_dataloader()
        _compute_rlct(tb_logger.log_dir, train_loader, module.random_seed, tb_logger)

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


def _compute_rlct(
    log_dir: str,
    train_loader: torch.utils.data.DataLoader,
    random_seed: int,
    evaluation_logger: TensorBoardLogger,
) -> None:
    """
    Computes the RLCT and logs it using TensorBoard.

    Args:
    log_dir (str): Directory to read logs from for analysis.
    train_loader (torch.utils.data.DataLoader): DataLoader to use for re-evaluating the model.
    random_seed (int): Seed used for reproducibility.
    evaluation_logger (TensorBoardLogger): Logger used for recording analysis results.
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    rlct_steps = set()
    if "rlct" in ea.scalars.Keys():
        for scalar_event in ea.Scalars("rlct"):
            rlct_steps.add(scalar_event.step)

    for checkpoint, step, _ in _get_checkpoints_info(log_dir):
        if step - 1 not in rlct_steps:
            module = MNISTModule.load_from_checkpoint(checkpoint)
            rlct = models.evaluate_rlct(
                module.model, module.num_classes, train_loader, random_seed, DEVICE
            )
            evaluation_logger.log_metrics({"rlct": rlct}, step - 1)
        else:
            print(f"RLCT already calculated for step {step - 1}")


if __name__ == "__main__":
    batch_size = 512
    max_epochs = 50

    hidden_layer_sizes = [512, 256, 128, 64]
    aux_task_weights = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]

    for hidden_layer_size in hidden_layer_sizes:
        for aux_task_weight in aux_task_weights:
            for random_seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                hidden_layers = hidden_layer_size if isinstance(hidden_layer_size, list) else [hidden_layer_size]
                activation_target = f"activation{len(hidden_layers)}" # last hidden layer
                if aux_task_weight == 0.0:
                    model_name = f"{hidden_layer_size}_baseline"
                    module = MNISTModule(
                        dataset_name="MNISTDataModule",
                        dataset_config={
                            "batch_size": batch_size,
                        },
                        model_name="MLP",
                        model_params={"hidden_layer_sizes": hidden_layers},
                        optimizer_name="SGD",
                        optimizer_config={"lr": 0.005, "momentum": 0.9, "nesterov": True},
                        loss_name="SelfModelingLoss",
                        loss_params={"aux_task_weight": aux_task_weight},
                        num_classes=10,
                        self_modeling_target_layers=[],
                        random_seed=random_seed,
                    )
                else:
                    model_name = f"{hidden_layer_size}_selfm_AW={aux_task_weight}"
                    module = MNISTModule(
                        dataset_name="MNISTDataModule",
                        dataset_config={
                            "batch_size": batch_size,
                        },
                        model_name="MLP",
                        model_params={"hidden_layer_sizes": hidden_layers},
                        optimizer_name="SGD",
                        optimizer_config={"lr": 0.005, "momentum": 0.9, "nesterov": True},
                        loss_name="SelfModelingLoss",
                        loss_params={"aux_task_weight": aux_task_weight},
                        num_classes=10,
                        self_modeling_target_layers=[activation_target],
                        random_seed=random_seed,
                    )
                v = run_experiment(
                    module,
                    max_epochs,
                    module.self_modeling_target_layers,
                    logger_suffix=model_name,
                    skip_rlct=True,
                )
                print(f"Finished training version: {v}")

