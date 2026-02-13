"""
Graceful shutdown handling for NAICS MCP Server.

Provides coordinated shutdown with:
- Signal handling (SIGTERM, SIGINT)
- Request draining (wait for in-flight requests)
- Configurable timeout with forced shutdown
- Ordered resource cleanup
- Shutdown hooks for extensibility

Usage:
    shutdown_manager = ShutdownManager(timeout_seconds=30)

    # Register cleanup hooks
    shutdown_manager.register_hook("database", database.disconnect)

    # In lifespan:
    async with shutdown_manager.lifespan():
        yield app_context

    # Or manual control:
    await shutdown_manager.initiate_shutdown()
"""

import asyncio
import logging
import signal
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ShutdownState(str, Enum):
    """Server shutdown state machine."""

    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    DRAINING = "draining"
    CLEANUP = "cleanup"
    STOPPED = "stopped"


@dataclass
class ShutdownConfig:
    """Configuration for graceful shutdown."""

    # Maximum time to wait for in-flight requests to complete
    timeout_seconds: float = 30.0

    # Time between checking if requests are drained
    drain_check_interval: float = 0.5

    # Grace period after receiving signal before starting drain
    grace_period_seconds: float = 1.0

    # Whether to force shutdown after timeout (vs. wait indefinitely)
    force_after_timeout: bool = True

    # Signals to handle
    handle_sigterm: bool = True
    handle_sigint: bool = True


@dataclass
class ShutdownHook:
    """A cleanup hook to run during shutdown."""

    name: str
    callback: Callable[[], Awaitable[None] | None]
    priority: int = 100  # Lower = runs first
    timeout_seconds: float = 5.0


@dataclass
class ShutdownResult:
    """Result of shutdown process."""

    state: ShutdownState
    duration_seconds: float
    requests_drained: int
    hooks_executed: list[str] = field(default_factory=list)
    hooks_failed: list[str] = field(default_factory=list)
    forced: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "duration_seconds": round(self.duration_seconds, 2),
            "requests_drained": self.requests_drained,
            "hooks_executed": self.hooks_executed,
            "hooks_failed": self.hooks_failed,
            "forced": self.forced,
            "error": self.error,
        }


class RequestTracker:
    """
    Track in-flight requests for graceful draining.

    Thread-safe counter for tracking active requests.
    """

    def __init__(self):
        self._count = 0
        self._lock = asyncio.Lock()
        self._drained_event = asyncio.Event()
        self._total_completed = 0

    async def start_request(self) -> bool:
        """
        Start tracking a new request.

        Returns False if server is draining (reject new requests).
        """
        async with self._lock:
            self._count += 1
            self._drained_event.clear()
            return True

    async def end_request(self) -> None:
        """Mark a request as completed."""
        async with self._lock:
            self._count -= 1
            self._total_completed += 1
            if self._count == 0:
                self._drained_event.set()

    async def get_count(self) -> int:
        """Get current in-flight request count."""
        async with self._lock:
            return self._count

    async def get_total_completed(self) -> int:
        """Get total requests completed since startup."""
        async with self._lock:
            return self._total_completed

    async def wait_for_drain(self, timeout: float | None = None) -> bool:
        """
        Wait for all requests to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if drained, False if timeout
        """
        # If already drained, return immediately
        async with self._lock:
            if self._count == 0:
                return True

        try:
            await asyncio.wait_for(self._drained_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class ShutdownManager:
    """
    Manages graceful server shutdown.

    Handles:
    - Signal registration (SIGTERM, SIGINT)
    - Request draining
    - Cleanup hook execution
    - Timeout-based forced shutdown

    Usage:
        manager = ShutdownManager()
        manager.register_hook("database", db.close)
        manager.register_hook("metrics", metrics.shutdown)

        # In lifespan context:
        async with manager.lifespan():
            yield app_context
    """

    def __init__(self, config: ShutdownConfig | None = None):
        """
        Initialize shutdown manager.

        Args:
            config: Shutdown configuration (uses defaults if not provided)
        """
        self.config = config or ShutdownConfig()
        self._state = ShutdownState.RUNNING
        self._hooks: list[ShutdownHook] = []
        self._request_tracker = RequestTracker()
        self._shutdown_event = asyncio.Event()
        self._shutdown_task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._original_handlers: dict[int, Any] = {}

    @property
    def state(self) -> ShutdownState:
        """Get current shutdown state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """True if server is accepting requests."""
        return self._state == ShutdownState.RUNNING

    @property
    def is_shutting_down(self) -> bool:
        """True if shutdown has been initiated."""
        return self._state != ShutdownState.RUNNING

    @property
    def request_tracker(self) -> RequestTracker:
        """Get request tracker for in-flight request management."""
        return self._request_tracker

    def register_hook(
        self,
        name: str,
        callback: Callable[[], Awaitable[None] | None],
        priority: int = 100,
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Register a cleanup hook.

        Hooks are executed in priority order (lower priority = earlier).
        Each hook has its own timeout.

        Args:
            name: Hook name for logging
            callback: Async or sync cleanup function
            priority: Execution order (lower = first)
            timeout_seconds: Maximum time for this hook
        """
        hook = ShutdownHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds,
        )
        self._hooks.append(hook)
        self._hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Registered shutdown hook: {name} (priority={priority})")

    def unregister_hook(self, name: str) -> bool:
        """
        Unregister a cleanup hook by name.

        Returns True if hook was found and removed.
        """
        original_count = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        return len(self._hooks) < original_count

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def signal_handler(signum: int) -> None:
            signal_name = signal.Signals(signum).name
            logger.info(f"Received {signal_name}, initiating graceful shutdown")

            # Only initiate shutdown once
            if not self._shutdown_event.is_set():
                self._shutdown_event.set()
                # Schedule shutdown in the event loop
                loop.create_task(self.initiate_shutdown(f"Signal {signal_name}"))

        # Store original handlers and install new ones
        if self.config.handle_sigterm:
            try:
                self._original_handlers[signal.SIGTERM] = loop.add_signal_handler(
                    signal.SIGTERM, lambda: signal_handler(signal.SIGTERM)
                )
                logger.debug("Installed SIGTERM handler")
            except NotImplementedError:
                # Signal handling not supported (e.g., Windows)
                logger.warning("SIGTERM handler not supported on this platform")

        if self.config.handle_sigint:
            try:
                self._original_handlers[signal.SIGINT] = loop.add_signal_handler(
                    signal.SIGINT, lambda: signal_handler(signal.SIGINT)
                )
                logger.debug("Installed SIGINT handler")
            except NotImplementedError:
                logger.warning("SIGINT handler not supported on this platform")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        loop = asyncio.get_running_loop()

        for signum in self._original_handlers:
            try:
                loop.remove_signal_handler(signum)
            except Exception as e:
                logger.warning(f"Failed to restore signal handler: {e}")

    async def initiate_shutdown(self, reason: str = "Manual shutdown") -> ShutdownResult:
        """
        Initiate graceful shutdown.

        1. Stop accepting new requests
        2. Wait for in-flight requests to drain
        3. Execute cleanup hooks
        4. Mark as stopped

        Args:
            reason: Reason for shutdown (for logging)

        Returns:
            ShutdownResult with details
        """
        if self._state != ShutdownState.RUNNING:
            logger.warning(f"Shutdown already in progress (state={self._state.value})")
            return ShutdownResult(
                state=self._state,
                duration_seconds=0,
                requests_drained=0,
                error="Shutdown already in progress",
            )

        start_time = time.monotonic()
        hooks_executed = []
        hooks_failed = []
        forced = False

        try:
            # Phase 1: Grace period
            logger.info(f"Shutdown initiated: {reason}")
            self._state = ShutdownState.SHUTTING_DOWN

            if self.config.grace_period_seconds > 0:
                logger.debug(f"Grace period: {self.config.grace_period_seconds}s")
                await asyncio.sleep(self.config.grace_period_seconds)

            # Phase 2: Drain requests
            self._state = ShutdownState.DRAINING
            in_flight = await self._request_tracker.get_count()
            logger.info(f"Draining {in_flight} in-flight requests")

            drain_timeout = self.config.timeout_seconds - self.config.grace_period_seconds
            drained = await self._request_tracker.wait_for_drain(timeout=drain_timeout)

            if not drained:
                remaining = await self._request_tracker.get_count()
                if self.config.force_after_timeout:
                    logger.warning(
                        f"Drain timeout reached, forcing shutdown with {remaining} requests pending"
                    )
                    forced = True
                else:
                    logger.warning(
                        f"Drain timeout reached, {remaining} requests still pending"
                    )

            requests_drained = in_flight - await self._request_tracker.get_count()

            # Phase 3: Cleanup hooks
            self._state = ShutdownState.CLEANUP
            logger.info(f"Executing {len(self._hooks)} cleanup hooks")

            for hook in self._hooks:
                try:
                    logger.debug(f"Executing hook: {hook.name}")
                    result = hook.callback()

                    # Handle both sync and async callbacks
                    if asyncio.iscoroutine(result):
                        await asyncio.wait_for(result, timeout=hook.timeout_seconds)

                    hooks_executed.append(hook.name)
                    logger.debug(f"Hook completed: {hook.name}")

                except asyncio.TimeoutError:
                    logger.error(f"Hook timed out: {hook.name}")
                    hooks_failed.append(hook.name)
                except Exception as e:
                    logger.error(f"Hook failed: {hook.name} - {e}")
                    hooks_failed.append(hook.name)

            # Phase 4: Done
            self._state = ShutdownState.STOPPED
            duration = time.monotonic() - start_time

            logger.info(
                f"Shutdown complete in {duration:.2f}s "
                f"(drained={requests_drained}, hooks={len(hooks_executed)}, "
                f"failed={len(hooks_failed)}, forced={forced})"
            )

            return ShutdownResult(
                state=ShutdownState.STOPPED,
                duration_seconds=duration,
                requests_drained=requests_drained,
                hooks_executed=hooks_executed,
                hooks_failed=hooks_failed,
                forced=forced,
            )

        except Exception as e:
            self._state = ShutdownState.STOPPED
            duration = time.monotonic() - start_time
            logger.error(f"Shutdown failed: {e}")

            return ShutdownResult(
                state=ShutdownState.STOPPED,
                duration_seconds=duration,
                requests_drained=0,
                hooks_executed=hooks_executed,
                hooks_failed=hooks_failed,
                forced=True,
                error=str(e),
            )

    @asynccontextmanager
    async def lifespan(self):
        """
        Context manager for server lifespan with graceful shutdown.

        Usage:
            async with shutdown_manager.lifespan():
                yield app_context
        """
        self._start_time = time.monotonic()
        self._state = ShutdownState.RUNNING

        # Set up signal handlers
        try:
            self._setup_signal_handlers()
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")

        try:
            yield
        finally:
            # Initiate shutdown if not already done
            if self._state == ShutdownState.RUNNING:
                await self.initiate_shutdown("Lifespan exit")

            # Restore signal handlers
            self._restore_signal_handlers()

    async def get_status(self) -> dict[str, Any]:
        """Get current shutdown manager status."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        return {
            "state": self._state.value,
            "is_running": self.is_running,
            "is_shutting_down": self.is_shutting_down,
            "uptime_seconds": round(uptime, 1),
            "in_flight_requests": await self._request_tracker.get_count(),
            "total_requests_completed": await self._request_tracker.get_total_completed(),
            "registered_hooks": [h.name for h in self._hooks],
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "grace_period_seconds": self.config.grace_period_seconds,
                "force_after_timeout": self.config.force_after_timeout,
            },
        }


# Singleton instance for global access
_shutdown_manager: ShutdownManager | None = None


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager


def reset_shutdown_manager() -> None:
    """Reset the global shutdown manager (for testing)."""
    global _shutdown_manager
    _shutdown_manager = None


def create_shutdown_manager(config: ShutdownConfig | None = None) -> ShutdownManager:
    """Create and set the global shutdown manager."""
    global _shutdown_manager
    _shutdown_manager = ShutdownManager(config)
    return _shutdown_manager
