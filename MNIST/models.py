import torch
import os
import random
import sys
import lightning as L
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import Tensor
from devinterp.slt import estimate_learning_coeff
from devinterp.optim import SGLD
from torch.multiprocessing import Process, Queue, set_start_method
from typing import List, Any, Optional, Dict, Tuple, Union

set_start_method("spawn", force=True)


def run_rlct_in_process(
    queue: Queue,
    model: nn.Module,
    num_classes: int,
    train_loader: torch.utils.data.DataLoader,
    seed: Optional[int],
    device: str,
) -> None:
    """
    Runs the RLCT estimation in a separate process and puts the result into a multiprocessing queue.

    Args:
    queue (Queue): The queue to put the RLCT estimate into.
    model (nn.Module): The neural network model.
    num_classes (int): Number of classes in the output layer of the model.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    seed (Optional[int]): Random seed for reproducibility.
    device (str): Device to run the computation on ('cpu' or 'cuda').
    """
    torch.cuda.empty_cache()
    model = ClassificationOnlyModel(model, num_classes).to(device)
    model.eval()

    rlct_estimate = estimate_learning_coeff(
        model,
        train_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_kwargs=dict(
            lr=1e-5,
            noise_level=1.0,
            localization=100.0,
        ),
        sampling_method=SGLD,
        num_chains=1,
        num_draws=400,
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        device=device,
        seed=seed,
    )
    queue.put(rlct_estimate)


def evaluate_rlct(
    model: nn.Module,
    num_classes: int,
    train_loader: torch.utils.data.DataLoader,
    seed: Optional[int],
    device: str,
) -> float:
    """
    Evaluates the RLCT of a model by running a separate process.

    Args:
    model (nn.Module): The neural network model.
    num_classes (int): Number of classes for the model's output.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    seed (Optional[int]): Seed for random number generation to ensure reproducibility.
    device (str): Device to run the computation on ('cpu' or 'cuda').

    Returns:
    float: Estimated RLCT value.
    """
    queue = Queue()

    p = Process(
        target=run_rlct_in_process,
        args=(queue, model, num_classes, train_loader, seed, device),
    )
    p.start()
    p.join()

    rlct_estimate = queue.get()
    return rlct_estimate


class ClassificationOnlyModel(L.LightningModule):
    def __init__(self, trained_model: nn.Module, num_classes: int):
        """
        Initializes a model that restricts outputs to the specified number of classes.

        Args:
        trained_model (nn.Module): The pre-trained model whose output is to be restricted.
        num_classes (int): Number of classes for the output.
        """
        super(ClassificationOnlyModel, self).__init__()
        self.num_classes = num_classes
        with torch.no_grad():  # This was necessary to avoid a problem with deepcopying this network in the RLCT calculation
            self.trained_model = trained_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes input data and returns model predictions restricted to num_classes.

        Args:
        x (torch.Tensor): Input data.

        Returns:
        torch.Tensor: Output data restricted to num_classes.
        """
        # Forward pass through the layers up to the classification output layer
        x = self.trained_model(x)
        return x[
            :, : self.num_classes
        ]  # Grabs only the first `num_classes` elements of each prediction

class StrictClassificationOnlyModel(nn.Module):
    def __init__(self, trained_model: nn.Module, num_classes: int, cls_layer_name: str):
        super(StrictClassificationOnlyModel, self).__init__()
        self.trained_model = trained_model
        self.num_classes = num_classes

        cls_layer = getattr(trained_model, cls_layer_name)
        if isinstance(cls_layer, nn.Linear):
            in_features = cls_layer.in_features
            weights = cls_layer.weight.data
            bias = cls_layer.bias.data
            new_cls_layer = nn.Linear(in_features, num_classes)
            # copy the weights from the original model
            new_cls_layer.weight.data = weights[:num_classes]
            new_cls_layer.bias.data = bias[:num_classes]
            setattr(trained_model, cls_layer_name, new_cls_layer)
        else:
            raise ValueError("The last layer of the trained model is not fully connected.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trained_model(x)

class ASTBaseModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        optimizer_name: str,
        optimizer_config: Dict[str, Any],
        loss_name: str,
        loss_params: Dict[str, Any],
        num_classes: int,
        self_modeling_target_layers: List[str] = [],
        model_architecture: Optional[OrderedDict] = None,
    ):
        """
        Initializes a base module for self-modeling tasks in AST experiments.

        Args:
        model_name (str): Name of the model class.
        model_params (Dict[str, Any]): Parameters for model configuration.
        optimizer_name (str): Name of the optimizer class.
        optimizer_config (Dict[str, Any]): Configuration settings for the optimizer.
        num_classes (int): Number of classes for the classification task.
        aux_task_weight (float, optional): Weight for auxiliary tasks in the loss calculation.
        self_modeling_target_layers (List[str], optional): List of layer names for self-modeling.
        model_architecture (Optional[OrderedDict], optional): Ordered dictionary of model layers.
        """
        super().__init__()
        self.model_name = model_name
        self.model_params = model_params
        self.optimizer_name = optimizer_name
        self.optimizer_config = optimizer_config
        self.loss_name = loss_name
        self.loss_params = loss_params
        self.num_classes = num_classes
        self.self_modeling_target_layers = self_modeling_target_layers
        self.model_architecture = model_architecture

        self.save_hyperparameters()

        loss_class = getattr(sys.modules[__name__], self.loss_name)
        self.self_modeling_loss = loss_class(
            num_classes=num_classes, **self.loss_params
        )
        self.classification_loss = nn.CrossEntropyLoss()

    def register_target_layer_activations_hooks(self):
        """
        Registers forward hooks to capture activations of target layers for self-modeling.
        """
        self.target_layer_activations = {}  # Stores activations of the target layers
        self.layer_hooks = {}

        for target_layer_name in self.self_modeling_target_layers:
            target_layer_module = self._find_target_layer_module(target_layer_name)
            target_layer_module.register_forward_hook(self.forward_hook)
            self.layer_hooks[target_layer_module] = target_layer_name

    def _find_target_layer_module(self, target_layer_name: str) -> nn.Module:
        """
        Searches and returns the target layer module by its name.

        Args:
        target_layer_name (str): Name of the target layer to find.

        Returns:
        nn.Module: The module corresponding to the target layer.

        Raises:
        ValueError: If the target layer cannot be found within the model.
        """
        for name, layer in self.model.named_modules():
            if name == target_layer_name:
                return layer
        all_layers = [
            f"{name}({m.__class__.__name__})" for name, m in self.model.named_modules()
        ]
        print(f"Available layers: {all_layers}")
        raise ValueError(f"Could not find target layer {target_layer_name}")

    def forward_hook(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        """
        Hook to capture and store activations from target layers.

        Args:
        module (nn.Module): The module where the hook is attached.
        input (torch.Tensor): Input tensor to the module.
        output (torch.Tensor): Output tensor from the module.
        """
        flattened_output = output.view(output.size(0), -1).detach().clone()
        self.target_layer_activations[self.layer_hooks[module]] = flattened_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the model's output.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor of the model.
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Executes a training step using the provided batch.

        Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): The batch containing input features and labels.
        batch_idx (int): Index of the batch.

        Returns:
        torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        # if loss is a tuple, split up and log the parts
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("train_loss", loss)
            self.log("train_classification_loss", classification_loss)
            self.log("train_aux_loss", aux_task_loss)
        else:
            self.log("train_classification_loss", loss)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Executes a validation step using the provided batch.

        Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): The batch containing input features and labels.
        batch_idx (int): Index of the batch.

        Returns:
        torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("val_loss", loss)
            self.log("val_classification_loss", classification_loss)
            self.log("val_aux_loss", aux_task_loss)
        else:
            self.log("val_classification_loss", loss)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Executes a test step using the provided batch.

        Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): The batch containing input features and labels.
        batch_idx (int): Index of the batch.

        Returns:
        torch.Tensor: The computed loss and accuracy for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("test_loss", loss)
            self.log("test_classification_loss", classification_loss)
            self.log("test_aux_loss", aux_task_loss)
        else:
            self.log("test_classification_loss", loss)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self) -> Any:
        """
        Configures the optimizer according to the module's specifications.

        Returns:
        Any: Configured optimizer.
        """
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        return optimizer_class(params=self.model.parameters(), **self.optimizer_config)

    def calculate_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Calculates the loss for a given batch of predictions and labels. If self-modeling is enabled,
        it computes both classification and auxiliary losses.

        Args:
        y_hat (torch.Tensor): The predictions output by the model.
        y (torch.Tensor): The true labels corresponding to the inputs.

        Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            The computed loss, which may be a single tensor or a tuple of classification loss,
            auxiliary task loss, and total combined loss.
        """
        if self.self_modeling_target_layers:
            activations = torch.cat(
                [
                    self.target_layer_activations[layer]
                    for layer in self.self_modeling_target_layers
                ],
                dim=1,
            )
            return self.self_modeling_loss(y_hat, y, activations)
        else:
            return self.classification_loss(y_hat, y)

    def calculate_accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculates the accuracy of the predictions against the true labels.

        Args:
        y_hat (torch.Tensor): The model's predictions.
        y (torch.Tensor): The true labels.

        Returns:
        float: The accuracy as a percentage of correct predictions.
        """
        with torch.no_grad():
            y_pred = torch.argmax(y_hat[:, :10], dim=1)
            correct = (y_pred == y).sum().item()
            total = y.size(0)
            return correct / total

    def train_dataloader(self) -> Any:
        """
        Provides the training DataLoader.

        Returns:
        Any: The DataLoader for the training dataset.
        """
        return self.data_module.train_dataloader()

    def val_dataloader(self) -> Any:
        """
        Provides the validation DataLoader.

        Returns:
        Any: The DataLoader for the validation dataset.
        """
        return self.data_module.val_dataloader()

    def test_dataloader(self) -> Any:
        """
        Provides the test DataLoader.

        Returns:
        Any: The DataLoader for the test dataset.
        """
        return self.data_module.test_dataloader()


class FractionalMaxPool2d(nn.Module):
    def __init__(self, output_size: float = 1.41):
        """
        Initializes the FractionalMaxPool2d module which performs a 2D max pooling operation
        with fractional output size.

        Parameters:
        output_size (float): The inverse of the desired output size relative to the input size.
                             For example, if output_size is 1.41, the output size will be roughly
                             1/1.41 times the input size. Must be greater than 1.0.
        """
        super(FractionalMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FractionalMaxPool2d module.

        Parameters:
        x (torch.Tensor): The input tensor with shape (N, C, H, W).

        Returns:
        torch.Tensor: The output tensor after applying fractional max pooling.
        """
        out = F.adaptive_max_pool2d(
            x,
            output_size=(
                int(x.size(2) / self.output_size),
                int(x.size(3) / self.output_size),
            ),
        )
        return out


class SelfModelingLoss(nn.Module):
    """Custom loss function for the self-modeling variants that combines classification and MSE losses"""

    def __init__(
        self,
        num_classes: int,
        classification_weight: float = 1.0,
        aux_task_weight: float = 1.0,
    ):
        """
        Initializes the SelfModelingLoss which combines classification and mean squared error losses.

        Args:
        num_classes (int): Number of classes in the classification task.
        classification_weight (float): Weight for the classification loss component.
        aux_task_weight (float): Weight for the auxiliary task loss component (MSE loss).
        """
        super(SelfModelingLoss, self).__init__()
        self.classification_weight = classification_weight
        self.aux_task_weight = aux_task_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(
        self,
        combined_output: torch.Tensor,
        targets: torch.Tensor,
        hidden_activations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the custom loss which is a combination of classification and MSE losses.

        Args:
        combined_output (torch.Tensor): Output from the model, includes both classification outputs and predicted activations.
        targets (torch.Tensor): True labels for the classification task.
        hidden_activations (torch.Tensor): True activations from the self-modeling target layers.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the classification loss, auxiliary task loss, and total loss.
        """
        # Split the combined output into classification and auxiliary task parts
        classification_output, predicted_hidden_activations = torch.split(
            combined_output, [self.num_classes, hidden_activations.size(-1)], dim=1
        )

        # Classification Loss (Cross Entropy)
        classification_loss = self.cross_entropy_loss(classification_output, targets)

        # Auxiliary Task Loss (L2 Loss)
        aux_task_loss = F.mse_loss(predicted_hidden_activations, hidden_activations)

        # Combine the two losses with specified weights
        total_loss = (
            self.classification_weight * classification_loss
            + self.aux_task_weight * aux_task_loss
        )

        return classification_loss, aux_task_loss, total_loss


class AdaptiveSelfModelingLoss(nn.Module):
    """Custom loss function with adaptive weighting for classification and MSE losses"""

    def __init__(self, num_classes: int, max_aux_contribution: float = 0.5):
        """
        Initializes the AdaptiveSelfModelingLoss which combines classification and mean squared error losses.

        Args:
            num_classes (int): Number of classes in the classification task.
            max_aux_contribution (float): Maximum percentage contribution of the auxiliary loss to the total loss.
        """
        super(AdaptiveSelfModelingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.mean_class_loss = 0.0
        self.var_class_loss = 0.0
        self.mean_aux_loss = 0.0
        self.var_aux_loss = 0.0
        self.alpha = 0.9  # Smoothing factor for exponential moving average
        self.max_aux_contribution = (
            max_aux_contribution  # Maximum contribution of aux_loss to the total loss
        )

    def update_stats(
        self, mean_old: float, var_old: float, loss: float
    ) -> Tuple[float, float]:
        """Update running statistics for either loss type"""
        mean_new = self.alpha * mean_old + (1 - self.alpha) * loss
        var_new = self.alpha * var_old + (1 - self.alpha) * (loss - mean_new) ** 2
        return mean_new, var_new

    def forward(
        self, combined_output: Tensor, targets: Tensor, hidden_activations: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the adaptive loss which dynamically weights classification and MSE losses.

        Args:
            combined_output (Tensor): Output from the model, includes both classification outputs and predicted activations.
            targets (Tensor): True labels for the classification task.
            hidden_activations (Tensor): True activations from the self-modeling target layers.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing the classification loss, auxiliary task loss, and total loss.
        """
        # Split the combined output into classification and auxiliary task parts
        classification_output, predicted_hidden_activations = torch.split(
            combined_output, [self.num_classes, hidden_activations.size(-1)], dim=1
        )

        # Classification Loss (Cross Entropy)
        classification_loss = self.cross_entropy_loss(classification_output, targets)

        # Auxiliary Task Loss (MSE Loss)
        aux_task_loss = F.mse_loss(predicted_hidden_activations, hidden_activations)

        # Update statistics and calculate dynamic weights
        self.mean_class_loss, self.var_class_loss = self.update_stats(
            self.mean_class_loss, self.var_class_loss, classification_loss.item()
        )
        self.mean_aux_loss, self.var_aux_loss = self.update_stats(
            self.mean_aux_loss, self.var_aux_loss, aux_task_loss.item()
        )

        weight_class_loss = 1 / (self.var_class_loss + 1e-6)
        weight_aux_loss = 1 / (self.var_aux_loss + 1e-6)

        # Adjust aux_task_weight to not exceed maximum contribution percentage
        unadjusted_total_loss = (
            weight_class_loss * classification_loss + weight_aux_loss * aux_task_loss
        )
        max_aux_loss = self.max_aux_contribution * unadjusted_total_loss
        actual_aux_loss = min(weight_aux_loss * aux_task_loss, max_aux_loss)

        # Calculate total loss with adjusted aux contribution
        total_loss = weight_class_loss * classification_loss + actual_aux_loss

        return classification_loss, aux_task_loss, total_loss


class DeterministicAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a deterministic average pooling over the spatial dimensions (height and width).

        Args:
        x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
        torch.Tensor: Output tensor with dimensions reduced along H and W, resulting in shape (N, C, 1, 1).
        """
        return x.mean([2, 3], keepdim=True)


class DeterministicFractionalMaxPool2d(nn.Module):
    def __init__(self, reduction_factor: float = 1.41):
        """
        Initializes a deterministic max pooling layer with a defined reduction factor.

        Args:
        reduction_factor (float): Factor by which to reduce the spatial dimensions.
        """
        super(DeterministicFractionalMaxPool2d, self).__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies deterministic fractional max pooling to reduce the spatial dimensions of the input tensor.

        Args:
        x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
        torch.Tensor: Output tensor with reduced spatial dimensions based on the reduction factor.
        """
        stride = int(self.reduction_factor)
        kernel_size = stride  # Assumes input size % reduction_factor == 0
        return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)


class VGG16LikeModel(nn.Module):
    """
    VGG16-like convolutional neural network model designed for CIFAR-10 classification.

    The architecture is inspired by the original VGG16 model but adapted for the CIFAR-10 dataset's
    smaller image size and fewer classes. It includes multiple convolutional layers, activation functions,
    dropout layers, and fully connected layers.

    Attributes:
    output_dim (int): The number of classes for the output layer.

    Source code adapted from https://github.com/laplacetw/vgg-like-cifar10/
    """

    def __init__(
        self,
        output_dim: int = 10,
        use_deterministic_algorithms: bool = True,
        disabled_dropouts=None,
    ):
        """
        Initializes the VGG16LikeModel with the specified output dimension.

        Parameters:
        output_dim (int): The number of output classes for the final fully connected layer.
        use_deterministic_algorithms (bool): Flag to use deterministic pooling algorithms.
        disabled_dropouts (List[int]): List of indices of dropout layers to disable.
        """
        super(VGG16LikeModel, self).__init__()
        self.use_deterministic_algorithms = use_deterministic_algorithms
        self.disabled_dropouts = set(disabled_dropouts) if disabled_dropouts else set()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same")
        self.activation_conv1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.activation_conv2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.activation_conv3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.activation_conv4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, padding="same")
        self.activation_conv5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, padding="same")
        self.activation_conv6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(96, 128, kernel_size=3, padding="same")
        self.activation_conv7 = nn.LeakyReLU()
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.activation_conv8 = nn.LeakyReLU()
        self.conv9 = nn.Conv2d(128, 160, kernel_size=3, padding="same")
        self.activation_conv9 = nn.LeakyReLU()
        self.conv10 = nn.Conv2d(160, 160, kernel_size=3, padding="same")
        self.activation_conv10 = nn.LeakyReLU()
        self.conv11 = nn.Conv2d(160, 192, kernel_size=3, padding="same")
        self.activation_conv11 = nn.LeakyReLU()
        self.conv12 = nn.Conv2d(192, 192, kernel_size=3, padding="same")
        self.activation_conv12 = nn.LeakyReLU()
        self.conv13 = nn.Conv2d(192, 192, kernel_size=1, padding="same")
        self.activation_conv13 = nn.LeakyReLU()

        # bn was never used, but now it can't be removed if we want to load old checkpoints
        self.bn = nn.BatchNorm2d(192)

        if self.use_deterministic_algorithms:
            self.global_avg_pool = DeterministicAvgPool()
            self.fractional_max_pool_class = DeterministicFractionalMaxPool2d
        else:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fractional_max_pool_class = FractionalMaxPool2d
        self.fc1 = nn.Linear(192, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VGG16LikeModel.

        Parameters:
        x (torch.Tensor): The input tensor with shape (N, C, H, W), where
                          N is the batch size, C is the number of channels,
                          H is the height, and W is the width.

        Returns:
        torch.Tensor: The output tensor with shape (N, output_dim), where
                      output_dim is the number of classes.
        """
        x = self.activation_conv1(self.conv1(x))
        x = self.activation_conv2(self.conv2(x))
        if 1 not in self.disabled_dropouts:
            x = F.dropout(x, 0.3)

        x = self.activation_conv3(self.conv3(x))
        x = self.activation_conv4(self.conv4(x))
        x = self.fractional_max_pool_class()(x)
        if 2 not in self.disabled_dropouts:
            x = F.dropout(x, 0.35)

        x = self.activation_conv5(self.conv5(x))
        x = self.activation_conv6(self.conv6(x))
        x = self.fractional_max_pool_class()(x)
        if 3 not in self.disabled_dropouts:
            x = F.dropout(x, 0.35)

        x = self.activation_conv7(self.conv7(x))
        x = self.activation_conv8(self.conv8(x))
        x = self.fractional_max_pool_class()(x)
        if 4 not in self.disabled_dropouts:
            x = F.dropout(x, 0.4)

        x = self.activation_conv9(self.conv9(x))
        x = self.activation_conv10(self.conv10(x))
        x = self.fractional_max_pool_class()(x)
        if 5 not in self.disabled_dropouts:
            x = F.dropout(x, 0.45)

        x = self.activation_conv11(self.conv11(x))
        x = self.activation_conv12(self.conv12(x))
        x = self.fractional_max_pool_class()(x)
        if 6 not in self.disabled_dropouts:
            x = F.dropout(x, 0.5)

        x = self.activation_conv13(self.conv13(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes=[1024, 1024, 10],
        input_dim=28 * 28,
        in_channels=1,
        output_dim=10,
        activation="ReLU",
        with_bias=True,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim * in_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_dim = output_dim
        self.activation = activation
        self.with_bias = with_bias

        self.flatten = nn.Flatten()
        previous_size = self.input_dim
        for i, size in enumerate(self.hidden_layer_sizes):
            setattr(
                self, f"fc{i+1}", nn.Linear(previous_size, size, bias=self.with_bias)
            )
            activation_fn = (
                getattr(nn, self.activation)()
                if isinstance(self.activation, str)
                else self.activation
            )
            setattr(
                self, f"activation{i+1}", activation_fn
            )  # This was added so that the activation layers are explicitly named
            previous_size = size
        self.fc_final = nn.Linear(previous_size, self.output_dim, bias=self.with_bias)

    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.hidden_layer_sizes)):
            x = getattr(self, f"fc{i+1}")(x)
            activation_fn = getattr(self, f"activation{i+1}")
            x = activation_fn(x)
        x = self.fc_final(x)
        return x


def calculate_total_activation_size(
    model: nn.Module, dummy_input: torch.Tensor, self_modeling_target_layers: List[str]
) -> int:
    """
    Calculates the total size of the activations for specified layers within a model.

    Args:
    model (nn.Module): The model to evaluate.
    dummy_input (torch.Tensor): A sample input tensor for the model to perform a forward pass.
    self_modeling_target_layers (List[str]): List of model layer names for which to calculate activations size.

    Returns:
    int: The total size of activations for the specified layers.
    """
    activation_sizes = []

    def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Hook function to capture and calculate the size of activations.

        Args:
        module (nn.Module): The module from which activations are captured.
        input (torch.Tensor): The input tensor to the module.
        output (torch.Tensor): The output tensor from the module.
        """
        if output.dim() == 4:  # Convolutional layer output
            n_features = output.shape[1] * output.shape[2] * output.shape[3]
        elif output.dim() == 2:  # Fully connected layer output
            n_features = output.shape[1]
        else:
            raise ValueError("Unsupported layer output dimensionality")
        activation_sizes.append(n_features)

    hooks = []
    for name, layer in model.named_modules():
        if name in self_modeling_target_layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Perform a forward pass with the dummy input to trigger the hooks
    model.forward(dummy_input)

    total_size = sum(activation_sizes)
    for hook in hooks:
        hook.remove()

    return total_size


def get_model_architecture(model: nn.Module) -> OrderedDict:
    """
    Extracts and returns the architecture details of a model in an ordered dictionary.

    Args:
    model (nn.Module): The model from which to extract architecture details.

    Returns:
    OrderedDict: An ordered dictionary containing names and configurations of each layer in the model.
    """
    architecture = OrderedDict()
    for name, module in model.named_modules():
        # skip the root module
        if name == "":
            continue
        architecture[name] = repr(module)
    return architecture


def configure_random_seed(random_seed: Optional[int] = None) -> None:
    """
    Configures the random seed for reproducibility across numpy, python random, and torch operations.

    Args:
    random_seed (Optional[int]): The seed value to use for all random operations. If None, the random state is not altered.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
