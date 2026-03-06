# AGENTS.md - Codebase Guide for AI Agents

## Project Overview

`putils` is a Python utility library for debugging and profiling distributed AI model training workloads, with specialized support for NPU (Neural Processing Unit) and CUDA environments. It provides non-invasive instrumentation tools for tensor inspection, memory monitoring, performance profiling, and multi-process debugging.

---

## Build / Lint / Test Commands

### Installation

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a single test file
pytest tests/test_cache.py

# Run a single test function
pytest tests/test_cache.py::TestRolloutCache::test_cache_initialization

# Run tests with coverage
pytest --cov=. --cov-report=term-missing

# Skip slow tests
pytest -m "not slow"

# Run only tests requiring torch
pytest -m "requires_torch"

# Run tests requiring CUDA (skip if unavailable)
pytest -m "requires_cuda"

# Run tests requiring NPU (skip if unavailable)
pytest -m "requires_npu"
```

### Type Checking / Linting

No formal linting configuration exists. Use standard Python practices:

```bash
# If using mypy (optional)
mypy .

# If using ruff (optional)
ruff check .
```

---

## Code Style Guidelines

### Imports

Standard import order (enforced by convention):

```python
# 1. Standard library
import os
import time
import hashlib
from contextlib import contextmanager

# 2. Third-party packages
import torch
import torch.distributed as dist

# 3. Local imports (relative)
from .write2file import log2file
from .pprint import aprint, ifdebug
```

### Optional Dependencies Pattern

Handle optional imports gracefully with try/except:

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
```

Always check availability before using:

```python
if not TORCH_AVAILABLE:
    raise RuntimeError("torch is required for this operation")
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | `snake_case` | `get_device_name()`, `to_cpu_detached()` |
| Variables | `snake_case` | `cache_dir`, `mem_allocated` |
| Classes | `PascalCase` | `RolloutCache`, `TestTimerFunctions` |
| Constants | `UPPER_SNAKE_CASE` | `TORCH_AVAILABLE`, `TIMER_VERBOSE` |
| Private methods | `_leading_underscore` | `_hash_inputs()`, `_cache_path()` |

### Type Hints

Type hints are optional but encouraged for public APIs:

```python
# Preferred: with type hints
def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine."""
    ...

def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage."""
    ...

# Acceptable: without type hints for simple internal functions
def to_filename_safe(timestamp_str):
    return timestamp_str.replace("-", "").replace(":", "")
```

### Error Handling

1. **Raise descriptive errors for missing dependencies:**
   ```python
   if not TORCH_AVAILABLE:
       raise RuntimeError("torch is required to load cache")
   ```

2. **Use print() for user-facing messages:**
   ```python
   print(f"[Timer] '{name}' finished in {duration:.6f} seconds.")
   ```

3. **Use logging module for library-level logging:**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.warning(f"Device namespace '{device_name}' not found")
   ```

4. **Context managers should handle exceptions:**
   ```python
   @contextmanager
   def timer(name=""):
       try:
           yield
       finally:
           # Always cleanup
           ...
   ```

### Documentation

Use docstrings for public functions and classes:

```python
def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    
    This currently only supports CPU, CUDA, NPU.
    
    Returns:
        device: The device name as a string ('cuda', 'npu', or 'cpu').
    """
```

Comments can be in English or Chinese (both are acceptable in this codebase).

---

## Testing Guidelines

### Test Structure

- Use class-based organization: `class TestFeatureName:`
- One test class per source module
- Use descriptive test method names: `test_<what_is_being_tested>`

```python
class TestRolloutCache:
    """Test RolloutCache class."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test RolloutCache initializes correctly."""
        ...

    def test_cache_disabled(self):
        """Test cache behavior when disabled."""
        ...
```

### Fixtures

Define reusable fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def temp_cache_dir(self):
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### Conditional Tests

Use markers for tests with special requirements:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_tensor_operations(self):
    ...
```

### Mocking

Use `pytest-mock` (`mocker` fixture) or `unittest.mock`:

```python
@patch('timer.dist')
def test_timer_context_manager_basic(self, mock_dist):
    mock_dist.is_initialized.return_value = False
    ...
```

---

## Project-Specific Patterns

### Context Managers

Many utilities use context managers for clean resource handling:

```python
@contextmanager
def timer(name=""):
    # Setup
    start = time.perf_counter()
    try:
        yield
    finally:
        # Cleanup
        print(f"Duration: {time.perf_counter() - start}")

# Usage:
with timer("operation"):
    do_something()
```

### Distributed Training Awareness

Check `torch.distributed` status before operations:

```python
rank = dist.get_rank() if dist.is_initialized() else 0
if not print_rank0_only or rank == 0:
    print(f"Message from rank {rank}")
```

### Device Abstraction

Support both CUDA and NPU environments:

```python
from .device import get_device_name, get_torch_device

device_name = get_device_name()  # Returns 'cuda', 'npu', or 'cpu'
device_module = get_torch_device()  # Returns torch.cuda, torch.npu, etc.
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `cache.py` | Tensor caching with hash-based keys |
| `timer.py` | Timing utilities and decorators |
| `device.py` | Device detection (CUDA/NPU/CPU) |
| `debug.py` | VSCode debugger attachment utilities |
| `memory.py` | GPU/NPU memory monitoring |
| `profiling.py` | NPU profiling context managers |
| `pprint.py` | Pretty printing utilities |
| `tools/` | Standalone scripts (stack tracing, etc.) |
| `tests/` | Unit tests with pytest |

---

## Notes

- Python 3.7+ required
- Core dependencies: `portalocker`, `debugpy`
- Optional: `torch`, `torch_npu` (for GPU/NPU features)
- No formal formatting config - follow existing patterns in codebase
- Both English and Chinese comments are acceptable