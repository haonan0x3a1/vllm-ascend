import os
import threading
from typing import Any


MOONCAKE_FORCE_TCP_ENV = "MC_FORCE_TCP"
MOONCAKE_FORCE_TCP_VALUE = "1"
_host_transfer_engine_lock = threading.Lock()


def create_host_transfer_engine(hostname: str) -> Any:
    """Create a TCP-only Mooncake engine after the Ascend engine exists.

    Mooncake 0.3.12 selects TCP-only transport through ``MC_FORCE_TCP`` at
    engine initialization time.  Keep the process environment mutation inside
    the single-threaded connector initialization boundary and restore it before
    any transfer thread starts.
    """
    with _host_transfer_engine_lock:
        if MOONCAKE_FORCE_TCP_ENV in os.environ:
            raise RuntimeError(
                f"{MOONCAKE_FORCE_TCP_ENV} must be unset before initializing "
                "the Ascend and Host Mooncake engines."
            )
        try:
            from mooncake.engine import TransferEngine  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run vLLM with MooncakeConnector."
            ) from e

        os.environ[MOONCAKE_FORCE_TCP_ENV] = MOONCAKE_FORCE_TCP_VALUE
        try:
            engine = TransferEngine()
            ret_value = engine.initialize(hostname, "P2PHANDSHAKE", "tcp", "")
        finally:
            os.environ.pop(MOONCAKE_FORCE_TCP_ENV, None)
        if ret_value != 0:
            raise RuntimeError(f"Host TransferEngine initialization failed with ret_value: {ret_value}")
        return engine


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


global_te = GlobalTE()
