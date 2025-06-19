import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

from typing_extensions import Self

logger = logging.getLogger(__name__)


class AsyncDSPyManager:
    """
    Asynchronous manager for DSPy program calls using a thread pool.
    Provides async submit, result retrieval with timeout, and robust error handling.
    """

    def __init__(self: Self, max_workers: int = 4, default_timeout: float = 10.0) -> None:
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers: int = max_workers
        self.default_timeout: float = default_timeout
        logger.info(
            f"AsyncDSPyManager initialized with max_workers={max_workers}, "
            f"default_timeout={default_timeout}s"
        )

    async def __aenter__(self: Self) -> "AsyncDSPyManager":
        return self

    async def __aexit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.shutdown()

    def __enter__(self: Self) -> "AsyncDSPyManager":
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.shutdown()

    def __del__(self: Self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.shutdown()
        except Exception:
            logger.debug("AsyncDSPyManager shutdown failed during __del__", exc_info=True)

    async def submit(
        self: Self,
        dspy_callable: Callable[..., object],
        *args: object,
        timeout: float | None = None,
        **kwargs: object,
    ) -> asyncio.Future[object]:
        """
        Submit a DSPy callable to be executed asynchronously.

        Args:
            dspy_callable (Callable): The callable to execute.
            *args (object): Positional arguments for the callable.
            timeout (float | None, optional): Timeout for the callable.
            **kwargs (object): Keyword arguments for the callable.

        Returns:
            asyncio.Future: The future representing the asynchronous execution.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.error(
                "AsyncDSPyManager.submit called without a running event loop: %s",
                exc,
            )
            raise
        logger.info(
            f"Submitting DSPy call: {getattr(dspy_callable, '__name__', str(dspy_callable))}"
        )
        func: Callable[[], object] = partial(dspy_callable, *args, **kwargs)
        return loop.run_in_executor(self.executor, func)

    async def get_result(
        self: Self,
        future: asyncio.Future[object],
        default_value: object = None,
        timeout: float | None = None,
        dspy_callable: Callable[..., object] | None = None,
    ) -> object:
        """
        Await and retrieve the result of a submitted DSPy call.
        Returns the result, or default_value on timeout/error.
        """
        timeout_val = timeout if timeout is not None else self.default_timeout
        callable_name = (
            getattr(dspy_callable, "__name__", str(dspy_callable))
            if dspy_callable
            else "Unnamed DSPy call"
        )

        logger.debug(f"Awaiting result for {callable_name} with timeout {timeout_val:.2f}s.")
        try:
            result = await asyncio.wait_for(future, timeout=timeout_val)
            logger.info(f"{callable_name} completed successfully within timeout.")
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "AsyncDSPyManager.get_result: Timeout awaiting result for "
                f"{callable_name} after {timeout_val:.2f}s."
            )
            if not future.done():
                if future.cancel():
                    logger.debug(f"Cancelled underlying task for {callable_name} due to timeout.")
                else:
                    logger.debug(
                        "Failed to cancel underlying task for "
                        f"{callable_name} (might have already completed or started running)."
                    )
            return default_value
        except Exception as e:
            logger.error(f"Exception retrieving result for {callable_name}: {e}", exc_info=True)
            return default_value

    def shutdown(self: Self, wait: bool = True) -> None:
        """
        Gracefully shut down the thread pool executor.
        """
        logger.info("Shutting down AsyncDSPyManager executor.")
        self.executor.shutdown(wait=wait)

    async def run_with_timeout_async(
        self: Self,
        program_callable: Callable[..., object],
        *args: object,
        timeout: float | None = None,
        **kwargs: object,
    ) -> object:
        """
        Run a DSPy program asynchronously with a timeout.

        Args:
            program_callable (Callable): The callable to execute.
            *args (object): Positional arguments for the callable.
            timeout (float | None, optional): Timeout for the callable.
            **kwargs (object): Keyword arguments for the callable.

        Returns:
            object: The result of the executed program.
        """
        program_name = getattr(program_callable, "__name__", str(program_callable))
        effective_timeout = timeout if timeout is not None else self.default_timeout

        logger.debug(
            f"RUN_WITH_TIMEOUT: Executing {program_name} with timeout "
            f"{effective_timeout:.2f}s. Args: {args}, Kwargs: {kwargs}"
        )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(program_callable, *args, **kwargs), timeout=effective_timeout
            )
            logger.debug(
                f"RUN_WITH_TIMEOUT: {program_name} completed successfully within timeout."
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"RUN_WITH_TIMEOUT: Timeout for {program_name} after {effective_timeout:.2f}s."
            )
            return None
        except Exception as e:
            logger.error(f"RUN_WITH_TIMEOUT: Error for {program_name}: {e}", exc_info=True)
            return None


__all__ = ["AsyncDSPyManager"]
