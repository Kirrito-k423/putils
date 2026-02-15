"""Pytest fixtures for putils tests."""
import pytest


@pytest.fixture
def mock_torch(mocker):
    """Fixture to provide a mocked torch module."""
    mock = mocker.MagicMock()
    mock.is_tensor = mocker.MagicMock(return_value=False)
    mock.bfloat16 = mocker.MagicMock()
    mock.float16 = mocker.MagicMock()
    mock.float = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_distributed(mocker):
    """Fixture to provide mocked torch.distributed."""
    mock = mocker.MagicMock()
    mock.is_initialized = mocker.MagicMock(return_value=False)
    mock.get_rank = mocker.MagicMock(return_value=0)
    return mock
