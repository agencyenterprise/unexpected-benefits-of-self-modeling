import torch
import os
import random
import sys
import lightning as L
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from typing import List, Any, Optional, Dict, Tuple, Union


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
        batch_size: int,
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
        batch_size (int): Batch size for logging.
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
        self.batch_size = batch_size
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

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the model's output.

        Args:
        x (torch.Tensor): Input tensor.
        offsets (torch.Tensor): Offsets for the input tensor.

        Returns:
        torch.Tensor: Output tensor of the model.
        """
        return self.model(x, offsets)

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
        y, text, offsets = batch
        y_hat = self(text, offsets)
        loss = self.calculate_loss(y_hat, y)
        # if loss is a tuple, split up and log the parts
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("train_loss", loss, batch_size=self.batch_size)
            self.log(
                "train_classification_loss",
                classification_loss,
                batch_size=self.batch_size,
            )
            self.log("train_aux_loss", aux_task_loss, batch_size=self.batch_size)
        else:
            self.log("train_classification_loss", loss, batch_size=self.batch_size)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("train_accuracy", accuracy, batch_size=self.batch_size)
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
        y, text, offsets = batch
        y_hat = self(text, offsets)
        loss = self.calculate_loss(y_hat, y)
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("val_loss", loss, batch_size=self.batch_size)
            self.log(
                "val_classification_loss",
                classification_loss,
                batch_size=self.batch_size,
            )
            self.log("val_aux_loss", aux_task_loss, batch_size=self.batch_size)
        else:
            self.log("val_classification_loss", loss, batch_size=self.batch_size)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("val_accuracy", accuracy, batch_size=self.batch_size)

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
        y, text, offsets = batch
        y_hat = self(text, offsets)
        loss = self.calculate_loss(y_hat, y)
        if isinstance(loss, tuple):
            classification_loss, aux_task_loss, loss = loss
            self.log("test_loss", loss, batch_size=self.batch_size)
            self.log(
                "test_classification_loss",
                classification_loss,
                batch_size=self.batch_size,
            )
            self.log("test_aux_loss", aux_task_loss, batch_size=self.batch_size)
        else:
            self.log("test_classification_loss", loss, batch_size=self.batch_size)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log("test_accuracy", accuracy, batch_size=self.batch_size)
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
            y_pred = torch.argmax(y_hat[:, : self.num_classes], dim=1)
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


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes,
        vocab_size,
        embed_dim,
        output_dim,
        activation="ReLU",
        with_bias=True,
    ):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_dim = output_dim
        self.activation = activation
        self.with_bias = with_bias

        self.flatten = nn.Flatten() # not used
        previous_size = embed_dim
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
    
    def forward(self, text, offsets=None):
        if offsets is not None:
            x = self.embedding(text, offsets)
            for i in range(len(self.hidden_layer_sizes)):
                x = getattr(self, f"fc{i+1}")(x)
                activation_fn = getattr(self, f"activation{i+1}")
                x = activation_fn(x)
            x = self.fc_final(x)
            return x
        else:
            data = text
            num_data = data[-1].item()
            # remove the last element which is the number of data
            data = data[:-1]
            offsets = data[:num_data]
            text = data[num_data:]
            x = self.embedding(text, offsets)
            for i in range(len(self.hidden_layer_sizes)):
                x = getattr(self, f"fc{i+1}")(x)
                activation_fn = getattr(self, f"activation{i+1}")
                x = activation_fn(x)
            x = self.fc_final(x)
            return x

class TextClassificationOnlyModel(nn.Module):
    def __init__(self, trained_model: nn.Module, num_classes: int):
        super(TextClassificationOnlyModel, self).__init__()
        self.trained_model = trained_model
        self.num_classes = num_classes

        if isinstance(trained_model.fc_final, nn.Linear):
            in_features = trained_model.fc_final.in_features
            weights = trained_model.fc_final.weight.data
            bias = trained_model.fc_final.bias.data
            trained_model.fc_final = nn.Linear(in_features, num_classes)
            # copy the weights from the original model
            trained_model.fc_final.weight.data = weights[:num_classes]
            trained_model.fc_final.bias.data = bias[:num_classes]
        else:
            raise ValueError("The last layer of the trained model is not fully connected.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trained_model(x)

def calculate_total_activation_size(
    model: nn.Module, dummy_input: torch.Tensor, offsets: torch.Tensor, self_modeling_target_layers: List[str]
) -> int:
    """
    Calculates the total size of the activations for specified layers within a model.

    Args:
    model (nn.Module): The model to evaluate.
    dummy_input (torch.Tensor): A sample input tensor for the model to perform a forward pass.
    offsets (torch.Tensor): Offsets for the input tensor.
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
    model.forward(dummy_input, offsets)

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