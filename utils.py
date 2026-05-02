"""
Common utility functions for BoostingVAN.

This module contains shared utilities for logging, checkpointing, and configuration
that are used across different applications (SK, EA, etc.).
"""

import os
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch


def setup_dtype(dtype_str: str) -> tuple:
    """
    Set up numpy and torch dtypes based on string specification.

    Args:
        dtype_str: Either 'float32' or 'float64'

    Returns:
        Tuple of (numpy_dtype, torch_dtype)
    """
    if dtype_str == "float32":
        np_dtype = np.float32
        torch_dtype = torch.float32
    elif dtype_str == "float64":
        np_dtype = np.float64
        torch_dtype = torch.float64
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    return np_dtype, torch_dtype


def setup_seed(seed: int) -> int:
    """
    Set up random seeds for reproducibility.

    Args:
        seed: Random seed. If 0, generates a random seed.

    Returns:
        The actual seed used.
    """
    if seed == 0:
        seed = np.random.randint(1, 10**8)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def setup_numpy_torch():
    """Configure numpy and torch global settings."""
    np.seterr(all="raise")
    np.seterr(under="warn")
    np.set_printoptions(precision=8, linewidth=160)
    torch.set_printoptions(precision=8, linewidth=160)
    torch.backends.cudnn.benchmark = True


def ensure_dir(filename: str) -> None:
    """
    Ensure that the directory for a given filename exists.

    Args:
        filename: Path to a file whose parent directory should exist.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


class Logger:
    """
    A simple logger that writes to both stdout and file.

    Args:
        out_filename: Base path for output files. If None, only prints to stdout.
        no_stdout: If True, suppresses stdout output.
    """

    def __init__(self, out_filename: Optional[str] = None, no_stdout: bool = False):
        self.out_filename = out_filename
        self.no_stdout = no_stdout

    def log(self, message: str) -> None:
        """Log a message to file and/or stdout."""
        if self.out_filename:
            with open(f"{self.out_filename}.log", "a", newline="\n") as f:
                f.write(message + "\n")
        if not self.no_stdout:
            print(message)

    def err(self, message: str) -> None:
        """Log an error message to file and/or stdout."""
        if self.out_filename:
            with open(f"{self.out_filename}.err", "a", newline="\n") as f:
                f.write(message + "\n")
        if not self.no_stdout:
            print(message)

    def clear_log(self) -> None:
        """Clear the log file."""
        if self.out_filename:
            open(f"{self.out_filename}.log", "w").close()

    def clear_err(self) -> None:
        """Clear the error file."""
        if self.out_filename:
            open(f"{self.out_filename}.err", "w").close()


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Args:
        out_filename: Base path for output files.
        save_step: Interval for saving checkpoints. If 0, saving is disabled.
    """

    def __init__(self, out_filename: Optional[str] = None, save_step: int = 0):
        self.out_filename = out_filename
        self.save_step = save_step
        self.save_dir = f"{out_filename}_save" if out_filename else None

    def get_save_path(self, step: int) -> Optional[str]:
        """Get the path for saving a checkpoint at given step."""
        if not self.save_dir:
            return None
        return f"{self.save_dir}/{step}.state"

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            step: Current training step.
            model: The model to save.
            optimizer: Optional optimizer state to save.
            scheduler: Optional scheduler state to save.
            extra: Optional extra data to save.
        """
        if not self.save_dir or not self.save_step:
            return

        save_path = self.get_save_path(step)
        ensure_dir(save_path)

        state = {"step": step, "model_state_dict": model.state_dict()}

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra is not None:
            state.update(extra)

        torch.save(state, save_path)

    def load(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> dict:
        """
        Load a checkpoint.

        Args:
            step: Step number to load.
            model: The model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.

        Returns:
            The full checkpoint dictionary.
        """
        save_path = self.get_save_path(step)
        if not save_path or not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

        state = torch.load(save_path)
        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return state

    def get_last_checkpoint_step(self) -> int:
        """
        Get the step number of the last saved checkpoint.

        Returns:
            The step number, or -1 if no checkpoints exist.
        """
        if not self.save_dir or not self.save_step:
            return -1

        filename_list = glob(f"{self.save_dir}/*.state")
        if not filename_list:
            return -1

        steps = []
        for filename in filename_list:
            basename = os.path.basename(filename)
            try:
                step = int(basename.replace(".state", ""))
                steps.append(step)
            except ValueError:
                continue

        return max(steps) if steps else -1

    def clear_checkpoints(self) -> None:
        """Remove all saved checkpoints."""
        if not self.save_dir:
            return

        filename_list = glob(f"{self.save_dir}/*.state")
        for filename in filename_list:
            os.remove(filename)


def print_args(args: Any, print_fn: Callable[[str], None] = print) -> None:
    """
    Print all arguments.

    Args:
        args: Namespace object containing arguments.
        print_fn: Function to use for printing.
    """
    for key, value in vars(args).items():
        print_fn(f"{key} = {value}")
    print_fn("")


def get_device(cuda_id: int = 0) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        cuda_id: GPU ID to use. -1 for CPU.

    Returns:
        torch.device object.
    """
    if cuda_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_id}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

