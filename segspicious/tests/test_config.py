"""Tests for segspicious.config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from segspicious.config import _config, configure, get_config


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config after each test."""
    yield
    _config.repo_id = None
    _config._authenticated = False


class TestConfigure:
    def test_sets_repo_id(self):
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser"}
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            configure(repo_id="myorg/my-repo")

        assert _config.repo_id == "myorg/my-repo"
        assert _config._authenticated is True

    def test_warns_when_not_authenticated(self):
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("not logged in")
        with (
            patch("huggingface_hub.HfApi", return_value=mock_api),
            pytest.warns(match="Not logged in"),
        ):
            configure(repo_id="myorg/my-repo")

        assert _config.repo_id == "myorg/my-repo"
        assert _config._authenticated is False


class TestGetConfig:
    def test_raises_when_not_configured(self):
        with pytest.raises(RuntimeError, match="not configured"):
            get_config()

    def test_returns_config_when_set(self):
        _config.repo_id = "org/repo"
        cfg = get_config()
        assert cfg.repo_id == "org/repo"
