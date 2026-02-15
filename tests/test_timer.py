"""Tests for timer.py module."""
import time
import pytest
from unittest.mock import patch, MagicMock, mock_open


class TestTimerFunctions:
    """Test standalone timer utility functions."""

    def test_to_filename_safe(self):
        """Test timestamp to filename conversion."""
        from timer import to_filename_safe
        
        # Test basic timestamp conversion
        result = to_filename_safe("2024-01-15 14:30:45.123")
        assert "-" not in result
        assert ":" not in result
        assert " " not in result
        assert "." not in result
        assert "_" in result

    def test_get_formatted_time_format(self):
        """Test get_formatted_time returns expected format."""
        from timer import get_formatted_time
        
        result = get_formatted_time()
        # Should be in format: "YYYY-MM-DD HH:MM:SS.mmm"
        parts = result.split()
        assert len(parts) == 2
        
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3
        assert all(p.isdigit() for p in date_parts)
        
        time_parts = parts[1].split(":")
        assert len(time_parts) == 3

    def test_get_time_str(self):
        """Test get_time_str returns string."""
        from timer import get_time_str
        
        result = get_time_str()
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('timer.get_formatted_time')
    def test_mprint_format(self, mock_get_time):
        """Test mprint output format."""
        from timer import mprint
        
        mock_get_time.return_value = "2024-01-15 14:30:45.123"
        
        with patch('builtins.print') as mock_print:
            result = mprint("test message", level="INFO")
            
            mock_print.assert_called_once()
            assert "test message" in mock_print.call_args[0][0]
            assert "INFO" in mock_print.call_args[0][0]


class TestTimerContextManager:
    """Test timer context manager functionality."""

    @patch('timer.dist')
    @patch('timer.TIMER_VERBOSE', True)
    def test_timer_context_manager_basic(self, mock_dist):
        """Test basic timer context manager execution."""
        from timer import timer
        
        mock_dist.is_initialized.return_value = False
        
        with timer("test operation"):
            pass
        
        # Context manager should complete without error

    @patch('timer.dist')
    @patch('timer.TIMER_VERBOSE', True)
    def test_timer_with_exception(self, mock_dist):
        """Test timer handles exceptions gracefully."""
        from timer import timer
        
        mock_dist.is_initialized.return_value = False
        
        try:
            with timer("failing operation"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Exception should propagate
        
        # Timer should have finished despite exception


class TestTimerDecorator:
    """Test timer decorator functionality."""

    @patch('timer.dist')
    @patch('timer.TIMER_VERBOSE', True)
    def test_timer_decorator(self, mock_dist):
        """Test timer decorator on function."""
        from timer import timer_decorator
        
        mock_dist.is_initialized.return_value = False
        
        @timer_decorator("decorated_func")
        def sample_func():
            return 42
        
        result = sample_func()
        assert result == 42

    @patch('timer.dist')
    @patch('timer.TIMER_VERBOSE', True)
    def test_timer_decorator_with_args(self, mock_dist):
        """Test timer decorator with function arguments."""
        from timer import timer_decorator
        
        mock_dist.is_initialized.return_value = False
        
        @timer_decorator("func_with_args")
        def sample_func(a, b, c=None):
            return a + b + (c or 0)
        
        result = sample_func(1, 2, c=3)
        assert result == 6
