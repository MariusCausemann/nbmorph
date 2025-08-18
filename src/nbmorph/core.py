import numpy as np
import numba
from .mode import onlyzero_mode_box, onlyzero_mode_diamond
from .minmax import minimum_box, minimum_diamond
from .zero_edges import zero_label_edges_box, zero_label_edges_diamond
from .utils import cycle, border_update3D

@numba.njit(cache=True)
def dilate_labels_spherical(labels: np.ndarray, radius: int=1, 
                            struct_sequence:str="DDB") -> np.ndarray:
    """
    Performs fast, quasi-spherical multilabel dilation on a CPU.
    This approximates a spherical structuring element by alternating between a box (3x3x3)
    and a diamond (6-connected) kernel for each iteration.
    Args:
        labels: The input 3D labeled image (must be an integer type).
        radius: The dilation radius in voxels. Each iteration expands by one voxel.

    Returns:
        The dilated 3D labeled image.
    """
    # Use two buffers that are swapped (ping-pong) to propagate labels iteratively
    # We start by copying the original labels to both

    # Initial setup: The original data is the first "pong".
    pong = np.copy(labels)
    ping = np.zeros_like(pong)

    for s in cycle(struct_sequence, radius):
        # Determine which kernel to use for this iteration
        if s=="B":
            onlyzero_mode_box(pong, out=ping)
        elif s=="D":
            onlyzero_mode_diamond(pong, out=ping)
        border_update3D(ping)

        # --- Swap Buffers for the Next Iteration ---
        # The ping of this step becomes the pong for the next one.
        pong, ping = ping, pong
    return pong

@numba.njit(cache=True)
def erode_labels_spherical(labels: np.ndarray, radius: int=1, 
                           struct_sequence:str="DDB") -> np.ndarray:
    """
    Performs fast, quasi-spherical multilabel erosion on a CPU.

    Args:
        labels (np.ndarray): The input 3D labeled image.
        radius (int, optional): The erosion radius. Defaults to 1.

    Returns:
        np.ndarray: The eroded labeled image.
    """
    assert radius > 0
    pong = np.copy(labels)
    ping = np.empty_like(pong)

    if struct_sequence[0]=="D":
        zero_label_edges_diamond(pong, out=ping)
    elif struct_sequence[0]=="B":
        zero_label_edges_box(pong, out=ping)

    border_update3D(ping)

    if radius==1: 
        return ping
    
    for s in cycle(struct_sequence, radius)[1:]:
        if s=="B":
            minimum_box(ping, out=pong)
        elif s=="D":
            minimum_diamond(ping, out=pong)
        border_update3D(pong)
        ping, pong = pong, ping
    return ping

@numba.njit
def open_labels_spherical(labels: np.ndarray, radius: int=1, iterations:int=1) -> np.ndarray:
    """
    Performs a morphological opening on a labeled image.

    Args:
        labels (np.ndarray): The input 3D labeled image.
        radius (int): The radius of the spherical structuring element.
        iterations (int, optional): The number of iterations. Defaults to 1.

    Returns:
        np.ndarray: The opened labeled image.
    """
    for i in range(iterations):
        labels = erode_labels_spherical(labels, radius)
        labels = dilate_labels_spherical(labels, radius)
    return labels

@numba.njit
def close_labels_spherical(labels: np.ndarray, radius: int, iterations:int=1) -> np.ndarray:
    """
    Performs a morphological closing on a labeled image.

    Args:
        labels (np.ndarray): The input 3D labeled image.
        radius (int): The radius of the spherical structuring element.
        iterations (int, optional): The number of iterations. Defaults to 1.

    Returns:
        np.ndarray: The closed labeled image.
    """
    for i in range(iterations):
        labels = dilate_labels_spherical(labels, radius)
        labels = erode_labels_spherical(labels, radius)
    return labels

@numba.njit
def smooth_labels_spherical(labels: np.ndarray, radius: int, iterations:int=1) -> np.ndarray:
    """
    Performs a morphological smoothing on a labeled image.

    Args:
        labels (np.ndarray): The input 3D labeled image.
        radius (int): The radius of the spherical structuring element.
        iterations (int, optional): The number of iterations. Defaults to 1.

    Returns:
        np.ndarray: The smoothed labeled image.
    """
    for i in range(iterations):
        labels = open_labels_spherical(labels, radius)
        labels = close_labels_spherical(labels, radius)
    return labels
