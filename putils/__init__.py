"""putils - Development AI model utilities package."""

__version__ = "0.1.0"

# Define public API - modules can be imported with: from putils import cache, timer, device
# Note: Modules are not imported at package level to avoid import errors when optional
# dependencies (torch, torch_npu) are not installed. Use lazy imports or import directly.
__all__ = [
    "accuracy",
    "burn",
    "cache",
    "debug",
    "device",
    "memory",
    "perf",
    "pprint",
    "profiling",
    "saver",
    "timer",
    "trace_manager",
    "write2file",
]


def __getattr__(name):
    """Lazy import modules when accessed."""
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")