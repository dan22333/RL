"""Neural-network utilities shared across the policy and baseline models.

Think of this file as the toolbox both networks reach into. It answers three
questions for the rest of the codebase:

    1. build_mlp     -- "give me a fresh neural network of this shape"
    2. np2torch      -- "turn this numpy array into a tensor I can feed the net"
    3. device        -- "which hardware (CPU/GPU) should everything live on?"

Keeping these in one place means the policy and the baseline can't accidentally
disagree about how networks are built or where tensors are placed.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def _select_device() -> torch.device:
    """Decide *once* which hardware every tensor and network should live on.

    Deep-learning math runs far faster on a GPU than a CPU, so we prefer one if
    it exists. We check, in order of speed:
        - "cuda": an NVIDIA GPU.
        - "mps":  Apple Silicon's GPU (Metal), on M1/M2/M3 Macs.
        - "cpu":  the fallback that always works.

    Running this at import time means the choice is made a single time and then
    reused everywhere via the module-level `device` below.

    NOTE: we deliberately pin to CPU here. The networks in this assignment are
    tiny MLPs and are never moved off the CPU (no model `.to(device)` calls), so
    sending *data* to an accelerator like Apple's MPS just creates a data-on-GPU /
    weights-on-CPU mismatch (a hard crash) — and, as benchmarked, these small nets
    actually run faster on CPU than MPS once GPU transfer overhead is counted.
    An NVIDIA CUDA box would still need the models moved too; add that if you port
    this to a GPU machine and place the networks on the device as well.
    """
    return torch.device("cpu")


# The one device the whole program uses. Other files import this and call
# `.to(device)` so models and data always end up on the same hardware.
device = _select_device()


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Callable[[], nn.Module] = nn.ReLU,
    output_gain: float = 0.01,
) -> nn.Sequential:
    """Build a plain feed-forward neural network (a "multi-layer perceptron").

    WHAT IT DOES, in plain terms: it stacks `n_layers` hidden layers, each with
    `size` neurons and a ReLU nonlinearity, then finishes with one *linear*
    output layer. Visually::

        input --> [Linear -> ReLU] --> [Linear -> ReLU] --> ... --> Linear --> output
                  \______________________ n_layers times ______________/

    WHY the last layer has no activation: the output must be free to be any real
    number. Depending on who calls this, the output represents action logits
    (discrete policy), action means (continuous policy), or a single state-value
    number (the baseline). A ReLU would clamp negatives to zero and break all of
    those.

    WHY the weird weight initialization: brand-new networks start with random
    weights, and the *scale* of that randomness matters a lot for RL stability.
    We use "orthogonal" initialization, the standard trick in policy-gradient
    code (see Engstrom et al. 2020, "Implementation Matters in Deep RL"):
        - Hidden layers get gain sqrt(2), the value tuned to pair with ReLU.
        - The output layer gets a *tiny* gain (output_gain) so the network
          starts out nearly neutral: a near-uniform policy / near-zero values,
          which makes early training calmer.

    Args:
        input_size:  Length of each input vector (e.g. the observation size).
        output_size: Length of each output vector (e.g. number of actions).
        n_layers:    How many hidden layers to stack (must be >= 1).
        size:        Neurons per hidden layer.
        activation:  The nonlinearity to use between hidden layers.
        output_gain: Init scale for the final layer (small = calm start).

    Returns:
        An `nn.Sequential` you can call like a function: `net(input_tensor)`.
    """
    # Fail loudly on nonsense shapes instead of producing a broken network.
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if min(input_size, output_size, size) < 1:
        raise ValueError(
            f"layer sizes must be >= 1, got input={input_size}, "
            f"output={output_size}, size={size}"
        )

    layers: list[nn.Module] = []
    in_features = input_size  # width feeding the next layer; grows/shrinks as we go
    for _ in range(n_layers):
        # Hidden block: shrink/expand to `size`, then apply the nonlinearity.
        layers.append(_linear(in_features, size, gain=np.sqrt(2)))
        layers.append(activation())
        in_features = size  # the next layer now receives `size` inputs
    # Output head: map to the final size, and crucially NO activation after it.
    layers.append(_linear(in_features, output_size, gain=output_gain))

    return nn.Sequential(*layers)


def _linear(in_features: int, out_features: int, gain: float) -> nn.Linear:
    """Create one fully-connected layer with our chosen initialization.

    This is a small helper so `build_mlp` doesn't repeat the same three lines for
    every layer. It builds a standard `nn.Linear`, then overwrites its random
    starting values with:
        - orthogonal weights scaled by `gain` (see build_mlp for the "why"), and
        - zero biases (a neutral, conventional starting point).
    The leading underscore marks it as internal — only this file should use it.
    """
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def np2torch(
    x: np.ndarray,
    cast_double_to_float: bool = True,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert a numpy array into a torch tensor, ready to feed a network.

    The environment and our data pipeline speak numpy, but the networks speak
    torch, so every array crossing that boundary passes through here. It does
    three small chores:

        1. Wrap the numpy array as a torch tensor (sharing memory when possible,
           so it's cheap).
        2. Downcast float64 -> float32 if asked. numpy often produces float64,
           but torch layers expect float32; skipping this causes dtype-mismatch
           errors, and Apple's MPS GPU rejects float64 entirely.
        3. Move the result onto the active `device` (GPU/CPU) so it lines up with
           the network's weights.

    NOTE the ordering: we cast to float32 *before* moving to the device. That's
    deliberate — moving a float64 tensor to an MPS GPU would crash outright, and
    transferring the smaller float32 tensor is also a bit faster.

    Args:
        x:                    The numpy array to convert.
        cast_double_to_float: Whether to downcast float64 to float32 (usually yes).
        target_device:        Where to place the tensor; defaults to `device`.

    Returns:
        A torch tensor on the requested device, ready for the network.
    """
    tensor = torch.as_tensor(x)  # numpy -> torch (still on CPU, original dtype)
    if cast_double_to_float and tensor.dtype == torch.float64:
        tensor = tensor.float()  # float64 -> float32, before any device move
    return tensor.to(target_device or device)  # finally, place it on GPU/CPU
