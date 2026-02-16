"""Tests for cache.py module."""
import os
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

# Import cache module to check torch availability
from cache import TORCH_AVAILABLE


class TestRolloutCache:
    """Test RolloutCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_initialization(self, temp_cache_dir):
        """Test RolloutCache initializes correctly."""
        from cache import RolloutCache
        
        cache = RolloutCache(temp_cache_dir, enabled=True, force_recompute=False)
        
        assert cache.cache_dir == temp_cache_dir
        assert cache.enabled is True
        assert cache.force_recompute is False
        assert os.path.exists(temp_cache_dir)

    def test_cache_disabled(self):
        """Test cache behavior when disabled."""
        from cache import RolloutCache
        
        cache = RolloutCache("/tmp/test_cache", enabled=False)
        
        assert cache.enabled is False

    def test_hash_inputs_with_strings(self):
        """Test _hash_inputs with string arguments."""
        from cache import RolloutCache
        
        cache = RolloutCache("/tmp/test", enabled=True)
        
        # Hash should be deterministic
        hash1 = cache._hash_inputs("test", "data")
        hash2 = cache._hash_inputs("test", "data")
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 40  # SHA-1 hex length

    def test_hash_inputs_different_values(self):
        """Test _hash_inputs produces different hashes for different values."""
        from cache import RolloutCache
        
        cache = RolloutCache("/tmp/test", enabled=True)
        
        hash1 = cache._hash_inputs("value1")
        hash2 = cache._hash_inputs("value2")
        
        assert hash1 != hash2

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('cache.torch')
    def test_hash_inputs_with_tensors(self, mock_torch):
        """Test _hash_inputs with tensor arguments."""
        from cache import RolloutCache
        
        cache = RolloutCache("/tmp/test", enabled=True)
        
        # Create mock tensor
        mock_tensor = MagicMock()
        mock_tensor.shape = [2, 3]
        mock_tensor.dtype = mock_torch.float32
        mock_tensor.device.type = "cpu"
        
        mock_torch.is_tensor.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float = MagicMock(return_value=mock_tensor)
        
        # Should not raise error
        result = cache._hash_inputs(mock_tensor)
        
        assert isinstance(result, str)


class TestCacheLoadSave:
    """Test cache load and save operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_nonexistent_key(self, temp_cache_dir):
        """Test loading non-existent cache key returns None."""
        from cache import RolloutCache
        
        cache = RolloutCache(temp_cache_dir, enabled=True)
        
        result = cache.load("nonexistent_key")
        
        assert result is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('cache.torch')
    def test_save_and_load(self, mock_torch, temp_cache_dir):
        """Test saving and loading cached data."""
        from cache import RolloutCache
        
        cache = RolloutCache(temp_cache_dir, enabled=True)
        
        test_data = {"key": "value", "number": 42}
        
        # Mock torch.save
        with patch('torch.save') as mock_save:
            cache.save("test_key", test_data)
            mock_save.assert_called_once()


class TestRolloutCacheContextManager:
    """Test rollout_cache context manager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rollout_cache_disabled(self):
        """Test context manager when cache is disabled."""
        from cache import rollout_cache, RolloutCache
        
        cache = RolloutCache("/tmp/test", enabled=False)
        
        with rollout_cache(cache, "input1", "input2") as result:
            assert result is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('cache.torch')
    def test_rollout_cache_miss(self, mock_torch, temp_cache_dir):
        """Test context manager on cache miss."""
        from cache import rollout_cache, RolloutCache
        
        cache = RolloutCache(temp_cache_dir, enabled=True, force_recompute=True)
        
        with rollout_cache(cache, "test_input") as result:
            # Should yield key string on cache miss
            assert isinstance(result, str)


class TestToCpuDetached:
    """Test to_cpu_detached utility function."""

    def test_to_cpu_detached_with_primitive(self):
        """Test to_cpu_detached with primitive types."""
        from cache import to_cpu_detached
        
        assert to_cpu_detached("string") == "string"
        assert to_cpu_detached(42) == 42
        assert to_cpu_detached(3.14) == 3.14
        assert to_cpu_detached(None) is None

    def test_to_cpu_detached_with_dict(self):
        """Test to_cpu_detached with dictionary."""
        from cache import to_cpu_detached
        
        input_dict = {"a": 1, "b": "string", "c": None}
        result = to_cpu_detached(input_dict)
        
        assert result == input_dict
        assert isinstance(result, dict)

    def test_to_cpu_detached_with_list(self):
        """Test to_cpu_detached with list."""
        from cache import to_cpu_detached
        
        input_list = [1, "string", 3.14, None]
        result = to_cpu_detached(input_list)
        
        assert result == input_list
        assert isinstance(result, list)

    def test_to_cpu_detached_with_tuple(self):
        """Test to_cpu_detached with tuple."""
        from cache import to_cpu_detached
        
        input_tuple = (1, "string", 3.14)
        result = to_cpu_detached(input_tuple)
        
        assert result == input_tuple
        assert isinstance(result, tuple)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch('cache.torch')
    def test_to_cpu_detached_with_tensor(self, mock_torch):
        """Test to_cpu_detached with tensor."""
        from cache import to_cpu_detached
        
        mock_tensor = MagicMock()
        mock_torch.is_tensor.return_value = True
        
        with patch('torch.is_tensor', return_value=True):
            # Create mock tensor behavior
            mock_tensor.detach.return_value.cpu.return_value = "detached_tensor"
            
            # Test would need actual tensor mocking setup
            # For now, just test structure
            result = to_cpu_detached(mock_tensor)
