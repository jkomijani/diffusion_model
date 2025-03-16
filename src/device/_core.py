# Copied from "https://github.com/jkomijani/normflow", 2024

import torch

import torch.distributed as dist


# =============================================================================
class DDP(torch.nn.parallel.DistributedDataParallel):
    # After wrapping a Module with DistributedDataParallel, the attributes of
    # the module (e.g. custom methods) became inaccessible. To access them,
    # a workaround is to use a subclass of DistributedDataParallel as here.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# =============================================================================
class ModelDeviceHandler:
    """
    A handler for managing device and distributed training setup in multi-GPU
    environments.

    This class helps manage distributed training by initializing and destroying
    the process group, setting seeds for reproducibility across processes, and
    moving model components to the appropriate device.
    """

    def __init__(self, model):
        """
        Initializes the ModelDeviceHandler instance.

        Args:
            model: The model that will be trained on multiple GPUs.
        """
        self._model = model
        self.world_size = 1
        self.rank = 0  # The global rank of the current process
        self.local_rank = 0  # The rank within the local node (GPU)

    def init_process_group(self, backend="nccl"):
        """
        Initializes the distributed process group for multi-GPU training.

        Args:
            backend (str): The backend for communication (e.g. 'nccl' for GPU).

        Sets the world size, rank, and local rank of the current process.
        """

        assert torch.cuda.is_available()

        # Initialize distributed backend
        dist.init_process_group(backend=backend)

        self.world_size = dist.get_world_size()

        gpus_per_node = torch.cuda.device_count()

        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank() % gpus_per_node

    def destroy_process_group(self):
        """
        Destroys the distributed process group and cleans up the resources.

        After calling this, no further distributed operations can be performed
        until re-initialization.
        """
        dist.destroy_process_group()

    def set_seed(self, seeds_list=None):
        """
        Sets the random seed for reproducibility across distributed processes.

        Each process will use a different seed based on its rank to ensure
        independent randomness.

        Parameters:
        -----------
        seeds_list : list, optional
            A list of seeds, one for each process. If None, no seed is set.
        """
        if seeds_list is not None:
            seed = seeds_list[self.rank]
            torch.manual_seed(seed)

    def to(self, *args, **kwargs):
        """
        Moves model components (score_net and prior) to the specified device.

        Args:
            *args: Positional arguments for the `.to()` method
                   (usually the device like 'cuda' or 'cpu').
            **kwargs: Keyword arguments for additional configuration
                   (e.g., dtype).
        """
        self._model.sdiffusion_process.core_func.to(*args, **kwargs)
        self._model.prior.to(*args, **kwargs)

    def ddp_wrapper(self):
        """
        Wraps the model in DistributedDataParallel (DDP) and moves it to the
        device (GPU) specified with the local rank.

        This method moves `score_func` and `prior` parts of the model to the
        specified GPU and wraps score_func w/ DDP to enable multi-GPU training.
        """

        device = torch.device(f"cuda:{self.local_rank}")

        # First, move the model (prior and score_func) to the specific GPU
        self._model.prior.to(device=device, dtype=None, non_blocking=False)
        self._model.diffusion_process.score_func.to(
            device=device, dtype=None, non_blocking=False
        )

        # Second, wrap score_func with DDP class
        self._model.diffusion_process.score_func = DDP(
            self._model.diffusion_process.score_func,
            device_ids=[device], output_device=device
        )

    def all_gather_into_tensor(self, x):
        if self.world_size == 1:
            return x
        else:
            out_shape = list(x.shape)
            out_shape[0] *= self.world_size
            out = torch.zeros(*out_shape, dtype=x.dtype, device=x.device)
            torch.distributed.all_gather_into_tensor(out, x)
            return out
