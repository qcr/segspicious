"""Global configuration for segspicious.

Call :func:`configure` once at the top of an experiment script to set the
Hugging Face Hub repository used for checkpoint storage::

    import segspicious
    segspicious.configure(repo_id="myorg/segspicious-checkpoints")

All subsequent calls to :func:`~segspicious.training.train`,
:func:`~segspicious.training.load`, and
:func:`~segspicious.training.train_or_load` will use this repository.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


@dataclass
class _Config:
    repo_id: str | None = field(default=None, repr=True)
    _authenticated: bool = field(default=False, repr=False)


_config = _Config()


def configure(repo_id: str) -> None:
    """Set the Hugging Face Hub repository for checkpoint storage.

    Also checks whether the user is authenticated with the Hub.
    Pulling from public repos works without authentication, but pushing
    new checkpoints will fail.

    Args:
        repo_id: A Hugging Face Hub repository ID,
            e.g. ``"myorg/segspicious-checkpoints"``.
    """
    from huggingface_hub import HfApi

    _config.repo_id = repo_id

    api = HfApi()
    try:
        api.whoami()
        _config._authenticated = True
    except Exception:
        _config._authenticated = False
        warnings.warn(
            "Not logged in to Hugging Face Hub. "
            "This is fine for pulling pre-trained checkpoints from public repos, "
            "but will fail if you try to push new checkpoints. "
            "Run `huggingface-cli login` to authenticate.",
            stacklevel=2,
        )


def get_config() -> _Config:
    """Return the current configuration, raising if not yet configured."""
    if _config.repo_id is None:
        raise RuntimeError(
            "segspicious is not configured. "
            "Call segspicious.configure(repo_id=...) at the start of your script."
        )
    return _config
