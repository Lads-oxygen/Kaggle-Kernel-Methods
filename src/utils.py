from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

def vector_to_image(
    v: np.ndarray,
    *,
    chw_shape: Tuple[int, int, int] = (3, 32, 32),
    normalise: bool = True,
) -> np.ndarray:
    """Reshape a flattened vector into an RGB image.

    Args:
        v: Flattened array of shape (C*H*W,).
        chw_shape: (C, H, W) target shape.
        normalise: If True, scales values to [0, 1] for display.

    Returns:
        Image array of shape (H, W, C).
    """

    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ValueError("v must be a 1D array")

    c, h, w = chw_shape
    expected = int(c * h * w)
    if arr.size != expected:
        raise ValueError(f"Expected v of length {expected}, got {arr.size}")

    img = arr.reshape(c, h, w).transpose(1, 2, 0)

    if normalise:
        mn = float(img.min())
        mx = float(img.max())
        img = (img - mn) / (mx - mn + 1e-12)

    return img


def show_vector_image(
    v: np.ndarray,
    *,
    title: Optional[str] = None,
    index: Optional[int] = None,
    chw_shape: Tuple[int, int, int] = (3, 32, 32),
    normalise: bool = True,
):
    """Display a flattened vector as an image using matplotlib.

    Args:
        v: Flattened array of shape (C*H*W,).
        title: Optional plot title.
        index: Optional index used in a default title.
        chw_shape: (C, H, W) target shape.
        normalise: If True, scales values to [0, 1] for display.

    Returns:
        The matplotlib Axes used for plotting.
    """

    img = vector_to_image(v, chw_shape=chw_shape, normalise=normalise)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")

    if title is None and index is not None:
        title = f"example {index}"
    if title:
        ax.set_title(title)

    return ax
