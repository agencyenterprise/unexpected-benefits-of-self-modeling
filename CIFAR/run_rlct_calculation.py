import os
import re
import json
import numpy as np
import torch
from torch import nn
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from collections import defaultdict
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.utils import optimal_temperature

import datasets
import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "{epoch:02d}-{step:02d}"
torch.multiprocessing.set_sharing_strategy('file_system')


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


def calculate_rlct(
    module: CIFAR10Module,
    version: str,
    logger_suffix: str,
) -> str:
    module.prepare_data()
    train_loader = module.data_module.train_dataloader()
    model = models.StrictClassificationOnlyModel(
        module.model, num_classes=10, cls_layer_name="linear"
    )

    rlct_dict = defaultdict(dict)
    try:
        with open(f"artifacts/CIFAR_rlcts.json") as f:
            rlct_dict.update(json.load(f))
    except FileNotFoundError:
        pass

    if logger_suffix in rlct_dict and version in rlct_dict[logger_suffix]:
        print(f"RLCT already calculated for {logger_suffix}/{version}")
        return logger_suffix

    llc_result = estimate_learning_coeff_with_summary(
        model,
        train_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_kwargs=dict(
            lr=1e-4,
            localization=100,
            temperature=optimal_temperature(train_loader),
        ),
        sampling_method=SGLD,
        num_chains=4,
        num_draws=400,
        # num_burnin_steps=200, # Does not work correctly in version 0.2.0 of devinterp
        online=True,
        verbose=False,
        seed=42,
        device=DEVICE,
    )

    llc_means = llc_result["llc/means"]
    llc = np.mean(llc_means[len(llc_means) // 2 :])

    try:
        with open(f"artifacts/CIFAR_rlcts.json", "r") as f:
            current_data = json.load(f)
    except FileNotFoundError:
        current_data = {}

    if logger_suffix not in current_data:
        current_data[logger_suffix] = {}
    current_data[logger_suffix][version] = str(llc)

    with open(f"artifacts/CIFAR_rlcts.json", "w") as f:
        json.dump(current_data, f, indent=4)

    return logger_suffix


def process_log_version(log: str, version: str, logs_dir: str, gpu_id: int) -> None:
    """
    Wrapper function to process each log and version directory.

    Args:
    log (str): Log directory name.
    version (str): Version directory name.
    logs_dir (str): Base logs directory.
    gpu_id (int): ID of the GPU to use.
    """
    torch.cuda.set_device(gpu_id)
    print(f"Running sweep for {log}/{version} on GPU {gpu_id}")
    last_checkpoint, _, _ = _get_checkpoints_info(
        os.path.join(logs_dir, log, version)
    )[-1]
    module = CIFAR10Module.load_from_checkpoint(last_checkpoint)
    print(f"Last checkpoint: {last_checkpoint}")
    calculate_rlct(
        module,
        version,
        logger_suffix=f"{log}",
    )
    print(f"Finished sweep for {log}/{version} on GPU {gpu_id}")


def assign_gpus_to_tasks(logs_dir: str, max_workers: int) -> List[Tuple[str, str, int]]:
    """
    Assign GPU IDs to tasks based on available logs and versions.

    Args:
    logs_dir (str): Base logs directory.
    max_workers (int): Maximum number of workers (threads).

    Returns:
    List[Tuple[str, str, int]]: List of tuples containing (log, version, gpu_id).
    """
    tasks = []
    gpu_id = 0

    for log in os.listdir(logs_dir):
        if not os.path.isdir(os.path.join(logs_dir, log)):
            continue

        for version in os.listdir(os.path.join(logs_dir, log)):
            if not os.path.isdir(os.path.join(logs_dir, log, version)):
                continue

            tasks.append((log, version, gpu_id))
            gpu_id = (gpu_id + 1) % torch.cuda.device_count()

    return tasks


if __name__ == "__main__":
    logs_dir = "artifacts/logs"
    max_workers = 2

    tasks = assign_gpus_to_tasks(logs_dir, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_log_version, log, version, logs_dir, gpu_id)
            for log, version, gpu_id in tasks
        ]

        for future in futures:
            future.result()
