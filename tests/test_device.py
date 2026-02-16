"""Tests for device.py module."""
import pytest
from unittest.mock import patch, MagicMock

# Import device module to check torch availability
from device import TORCH_AVAILABLE


class TestDeviceAvailability:
    """Test device availability detection functions."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('device.torch')
    def test_is_torch_npu_available_false_no_import(self, mock_torch):
        """Test NPU availability returns False when torch_npu not available."""
        from device import is_torch_npu_available
        
        # Simulate ImportError when trying to import torch_npu
        with patch('builtins.__import__', side_effect=ImportError()):
            result = is_torch_npu_available()
            assert result is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('device.torch')
    def test_is_cuda_available_property(self, mock_torch):
        """Test CUDA availability check."""
        from device import is_cuda_available
        
        # This tests the module-level variable was created correctly
        assert isinstance(is_cuda_available, bool)


class TestDeviceName:
    """Test device name related functions."""

    @patch('device.is_cuda_available', True)
    @patch('device.is_npu_available', False)
    def test_get_device_name_cuda(self):
        """Test get_device_name returns cuda when available."""
        from device import get_device_name
        
        result = get_device_name()
        assert result == "cuda"

    @patch('device.is_cuda_available', False)
    @patch('device.is_npu_available', True)
    def test_get_device_name_npu(self):
        """Test get_device_name returns npu when available."""
        from device import get_device_name
        
        result = get_device_name()
        assert result == "npu"

    @patch('device.is_cuda_available', False)
    @patch('device.is_npu_available', False)
    def test_get_device_name_cpu(self):
        """Test get_device_name returns cpu as fallback."""
        from device import get_device_name
        
        result = get_device_name()
        assert result == "cpu"


class TestVisibleDevices:
    """Test visible device environment variable functions."""

    @patch('device.is_cuda_available', True)
    @patch('device.is_npu_available', False)
    def test_get_visible_devices_keyword_cuda(self):
        """Test get_visible_devices_keyword returns CUDA keyword."""
        from device import get_visible_devices_keyword
        
        result = get_visible_devices_keyword()
        assert result == "CUDA_VISIBLE_DEVICES"

    @patch('device.is_cuda_available', False)
    @patch('device.is_npu_available', True)
    def test_get_visible_devices_keyword_npu(self):
        """Test get_visible_devices_keyword returns NPU keyword."""
        from device import get_visible_devices_keyword
        
        result = get_visible_devices_keyword()
        assert result == "ASCEND_RT_VISIBLE_DEVICES"


class TestTorchDevice:
    """Test torch device related functions."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('device.get_device_name')
    @patch('device.torch')
    def test_get_torch_device_success(self, mock_torch, mock_get_device):
        """Test get_torch_device returns correct torch device module."""
        from device import get_torch_device
        
        mock_get_device.return_value = "cuda"
        mock_torch.cuda = MagicMock()
        
        with patch.object(mock_torch, 'cuda', mock_torch.cuda):
            with patch.object(mock_torch, 'cuda', mock_torch.cuda, create=True):
                result = get_torch_device()
                # Result should be the cuda module or similar
                assert result is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('device.get_device_name')
    @patch('device.torch')
    def test_get_torch_device_fallback(self, mock_torch, mock_get_device):
        """Test get_torch_device falls back to torch.cuda."""
        from device import get_torch_device
        
        mock_get_device.return_value = "unknown_device"
        
        # Mock getattr to raise AttributeError
        with patch('builtins.getattr', side_effect=AttributeError()):
            result = get_torch_device()
            # Should return torch.cuda as fallback
            assert result is not None


class TestModuleLevelVariables:
    """Test module-level constants and variables."""

    def test_cuda_available_type(self):
        """Test is_cuda_available is a boolean."""
        from device import is_cuda_available
        
        assert isinstance(is_cuda_available, bool)

    def test_npu_available_type(self):
        """Test is_npu_available is a boolean."""
        from device import is_npu_available
        
        assert isinstance(is_npu_available, bool)

    def test_logger_exists(self):
        """Test logger is defined in module."""
        import device as device_module
        
        assert hasattr(device_module, 'logger')
