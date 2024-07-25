import os
import re
import json
import numpy as np
import torch
from torch import nn
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from collections import defaultdict
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.utils import optimal_temperature

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
    module: MNISTModule,
    version: str,
    logger_suffix: str,
) -> str:
    module.prepare_data()
    train_loader = module.data_module.train_dataloader()
    model = models.StrictClassificationOnlyModel(
        module.model, num_classes=10, cls_layer_name="fc_final"
    )

    rlct_dict = defaultdict(dict)
    try:
        with open(f"artifacts/MNIST_rlcts.json") as f:
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
        num_draws=1000,
        verbose=False,
        online=True,
        seed=42,
    )
    llc_means = llc_result["llc/means"]
    # take the mean of the second half of the samples
    llc = np.mean(llc_means[len(llc_means) // 2 :])

    try:
        with open(f"artifacts/MNIST_rlcts.json", "r") as f:
            current_data = json.load(f)
    except FileNotFoundError:
        current_data = {}

    if logger_suffix not in current_data:
        current_data[logger_suffix] = {}
    current_data[logger_suffix][version] = str(llc)

    with open(f"artifacts/MNIST_rlcts.json", "w") as f:
        json.dump(current_data, f, indent=4)

    return logger_suffix


def process_log_version(log: str, version: str, logs_dir: str) -> None:
    """
    Wrapper function to process each log and version directory.

    Args:
    log (str): Log directory name.
    version (str): Version directory name.
    logs_dir (str): Base logs directory.
    """
    print(f"Running sweep for {log}/{version}")
    last_checkpoint, _, _ = _get_checkpoints_info(
        os.path.join(logs_dir, log, version)
    )[-1]
    module = MNISTModule.load_from_checkpoint(last_checkpoint)
    print(f"Last checkpoint: {last_checkpoint}")
    calculate_rlct(
        module,
        version,
        logger_suffix=f"{log}",
    )
    print(f"Finished sweep for {log}/{version}")


if __name__ == "__main__":
    logs_dir = "artifacts/logs"
    max_workers = 8
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for log in os.listdir(logs_dir):
            if not os.path.isdir(os.path.join(logs_dir, log)):
                continue

            for version in os.listdir(os.path.join(logs_dir, log)):
                if not os.path.isdir(os.path.join(logs_dir, log, version)):
                    continue

                tasks.append(
                    executor.submit(process_log_version, log, version, logs_dir)
                )

        for task in tasks:
            task.result()
