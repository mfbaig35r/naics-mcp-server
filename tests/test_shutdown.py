"""
Tests for graceful shutdown module.
"""

import asyncio

import pytest

from naics_mcp_server.config import ShutdownConfig as PydanticShutdownConfig
from naics_mcp_server.core.shutdown import (
    RequestTracker,
    ShutdownConfig,
    ShutdownManager,
    ShutdownResult,
    ShutdownState,
    create_shutdown_manager,
    get_shutdown_manager,
    reset_shutdown_manager,
)


class TestRequestTracker:
    """Tests for RequestTracker."""

    @pytest.mark.asyncio
    async def test_start_and_end_request(self):
        """Test basic request tracking."""
        tracker = RequestTracker()

        assert await tracker.get_count() == 0

        await tracker.start_request()
        assert await tracker.get_count() == 1

        await tracker.end_request()
        assert await tracker.get_count() == 0

    @pytest.mark.asyncio
    async def test_multiple_requests(self):
        """Test tracking multiple concurrent requests."""
        tracker = RequestTracker()

        # Start 5 requests
        for _ in range(5):
            await tracker.start_request()

        assert await tracker.get_count() == 5

        # End 3 requests
        for _ in range(3):
            await tracker.end_request()

        assert await tracker.get_count() == 2

    @pytest.mark.asyncio
    async def test_total_completed(self):
        """Test total completed counter."""
        tracker = RequestTracker()

        # Complete several requests
        for _ in range(10):
            await tracker.start_request()
            await tracker.end_request()

        assert await tracker.get_total_completed() == 10
        assert await tracker.get_count() == 0

    @pytest.mark.asyncio
    async def test_wait_for_drain_immediate(self):
        """Test drain returns immediately when no requests."""
        tracker = RequestTracker()

        drained = await tracker.wait_for_drain(timeout=1.0)
        assert drained is True

    @pytest.mark.asyncio
    async def test_wait_for_drain_with_requests(self):
        """Test drain waits for requests to complete."""
        tracker = RequestTracker()

        await tracker.start_request()

        async def complete_request():
            await asyncio.sleep(0.1)
            await tracker.end_request()

        # Start completion in background
        asyncio.create_task(complete_request())

        # Wait for drain
        drained = await tracker.wait_for_drain(timeout=1.0)
        assert drained is True

    @pytest.mark.asyncio
    async def test_wait_for_drain_timeout(self):
        """Test drain timeout when requests don't complete."""
        tracker = RequestTracker()

        await tracker.start_request()

        # Short timeout, request won't complete
        drained = await tracker.wait_for_drain(timeout=0.05)
        assert drained is False

        # Cleanup
        await tracker.end_request()


class TestShutdownConfig:
    """Tests for ShutdownConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ShutdownConfig()

        assert config.timeout_seconds == 30.0
        assert config.drain_check_interval == 0.5
        assert config.grace_period_seconds == 1.0
        assert config.force_after_timeout is True
        assert config.handle_sigterm is True
        assert config.handle_sigint is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ShutdownConfig(
            timeout_seconds=60.0,
            grace_period_seconds=5.0,
            force_after_timeout=False,
        )

        assert config.timeout_seconds == 60.0
        assert config.grace_period_seconds == 5.0
        assert config.force_after_timeout is False


class TestPydanticShutdownConfig:
    """Tests for Pydantic ShutdownConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PydanticShutdownConfig()

        assert config.shutdown_timeout_seconds == 30.0
        assert config.grace_period_seconds == 1.0
        assert config.force_after_timeout is True

    def test_to_dict(self):
        """Test configuration serialization."""
        config = PydanticShutdownConfig()
        config_dict = config.to_dict()

        assert "shutdown_timeout_seconds" in config_dict
        assert "grace_period_seconds" in config_dict
        assert "force_after_timeout" in config_dict


class TestShutdownManager:
    """Tests for ShutdownManager."""

    def setup_method(self):
        """Reset global shutdown manager before each test."""
        reset_shutdown_manager()

    def teardown_method(self):
        """Clean up after each test."""
        reset_shutdown_manager()

    def test_initial_state(self):
        """Test manager starts in running state."""
        manager = ShutdownManager()

        assert manager.state == ShutdownState.RUNNING
        assert manager.is_running is True
        assert manager.is_shutting_down is False

    def test_register_hook(self):
        """Test registering cleanup hooks."""
        manager = ShutdownManager()

        async def cleanup():
            pass

        manager.register_hook("test_hook", cleanup)

        assert len(manager._hooks) == 1
        assert manager._hooks[0].name == "test_hook"

    def test_register_hook_priority(self):
        """Test hooks are sorted by priority."""
        manager = ShutdownManager()

        manager.register_hook("low", lambda: None, priority=100)
        manager.register_hook("high", lambda: None, priority=10)
        manager.register_hook("medium", lambda: None, priority=50)

        assert manager._hooks[0].name == "high"
        assert manager._hooks[1].name == "medium"
        assert manager._hooks[2].name == "low"

    def test_unregister_hook(self):
        """Test unregistering hooks."""
        manager = ShutdownManager()

        manager.register_hook("test", lambda: None)
        assert len(manager._hooks) == 1

        removed = manager.unregister_hook("test")
        assert removed is True
        assert len(manager._hooks) == 0

    def test_unregister_nonexistent_hook(self):
        """Test unregistering nonexistent hook."""
        manager = ShutdownManager()

        removed = manager.unregister_hook("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_initiate_shutdown(self):
        """Test basic shutdown initiation."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,  # Skip grace period for test
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        result = await manager.initiate_shutdown("Test")

        assert result.state == ShutdownState.STOPPED
        assert manager.state == ShutdownState.STOPPED
        assert result.forced is False

    @pytest.mark.asyncio
    async def test_shutdown_executes_hooks(self):
        """Test shutdown executes all hooks."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        hook_executed = []

        async def hook1():
            hook_executed.append("hook1")

        async def hook2():
            hook_executed.append("hook2")

        manager.register_hook("hook1", hook1, priority=10)
        manager.register_hook("hook2", hook2, priority=20)

        result = await manager.initiate_shutdown("Test")

        assert "hook1" in result.hooks_executed
        assert "hook2" in result.hooks_executed
        assert hook_executed == ["hook1", "hook2"]  # Priority order

    @pytest.mark.asyncio
    async def test_shutdown_handles_hook_failure(self):
        """Test shutdown continues after hook failure."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        async def failing_hook():
            raise RuntimeError("Hook failed")

        async def success_hook():
            pass

        manager.register_hook("failing", failing_hook, priority=10)
        manager.register_hook("success", success_hook, priority=20)

        result = await manager.initiate_shutdown("Test")

        assert "failing" in result.hooks_failed
        assert "success" in result.hooks_executed
        assert result.state == ShutdownState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_handles_hook_timeout(self):
        """Test shutdown handles slow hooks."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        async def slow_hook():
            await asyncio.sleep(10)  # Very slow

        manager.register_hook("slow", slow_hook, priority=10, timeout_seconds=0.1)

        result = await manager.initiate_shutdown("Test")

        assert "slow" in result.hooks_failed

    @pytest.mark.asyncio
    async def test_shutdown_drains_requests(self):
        """Test shutdown waits for in-flight requests."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            timeout_seconds=5.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        # Start some requests
        await manager.request_tracker.start_request()
        await manager.request_tracker.start_request()

        async def complete_requests():
            await asyncio.sleep(0.1)
            await manager.request_tracker.end_request()
            await asyncio.sleep(0.1)
            await manager.request_tracker.end_request()

        # Complete requests in background
        asyncio.create_task(complete_requests())

        result = await manager.initiate_shutdown("Test")

        assert result.requests_drained == 2
        assert result.forced is False

    @pytest.mark.asyncio
    async def test_shutdown_forces_after_timeout(self):
        """Test shutdown forces after timeout."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            timeout_seconds=0.1,  # Very short
            force_after_timeout=True,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        # Start a request that won't complete
        await manager.request_tracker.start_request()

        result = await manager.initiate_shutdown("Test")

        assert result.forced is True

        # Cleanup
        await manager.request_tracker.end_request()

    @pytest.mark.asyncio
    async def test_double_shutdown_rejected(self):
        """Test second shutdown is rejected."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        # First shutdown
        await manager.initiate_shutdown("First")

        # Second shutdown
        result = await manager.initiate_shutdown("Second")

        assert result.error == "Shutdown already in progress"

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status reporting."""
        config = ShutdownConfig(
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)
        manager._start_time = 0  # Set start time

        manager.register_hook("test", lambda: None)

        status = await manager.get_status()

        assert status["state"] == "running"
        assert status["is_running"] is True
        assert status["in_flight_requests"] == 0
        assert "test" in status["registered_hooks"]

    @pytest.mark.asyncio
    async def test_lifespan_context(self):
        """Test lifespan context manager."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        async with manager.lifespan():
            assert manager.is_running is True

        assert manager.state == ShutdownState.STOPPED

    @pytest.mark.asyncio
    async def test_sync_hook(self):
        """Test shutdown handles sync hooks."""
        config = ShutdownConfig(
            grace_period_seconds=0.0,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        sync_executed = []

        def sync_hook():
            sync_executed.append("sync")

        manager.register_hook("sync", sync_hook)

        result = await manager.initiate_shutdown("Test")

        assert "sync" in result.hooks_executed
        assert sync_executed == ["sync"]


class TestShutdownResult:
    """Tests for ShutdownResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = ShutdownResult(
            state=ShutdownState.STOPPED,
            duration_seconds=1.234,
            requests_drained=5,
            hooks_executed=["hook1", "hook2"],
            hooks_failed=["hook3"],
            forced=False,
        )

        result_dict = result.to_dict()

        assert result_dict["state"] == "stopped"
        assert result_dict["duration_seconds"] == 1.23
        assert result_dict["requests_drained"] == 5
        assert result_dict["hooks_executed"] == ["hook1", "hook2"]
        assert result_dict["hooks_failed"] == ["hook3"]
        assert result_dict["forced"] is False

    def test_to_dict_with_error(self):
        """Test result serialization with error."""
        result = ShutdownResult(
            state=ShutdownState.STOPPED,
            duration_seconds=0.5,
            requests_drained=0,
            error="Something went wrong",
        )

        result_dict = result.to_dict()

        assert result_dict["error"] == "Something went wrong"


class TestGlobalShutdownManager:
    """Tests for global shutdown manager functions."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_shutdown_manager()

    def teardown_method(self):
        """Clean up after each test."""
        reset_shutdown_manager()

    def test_get_shutdown_manager_creates_default(self):
        """Test get creates default manager."""
        manager = get_shutdown_manager()

        assert manager is not None
        assert manager.state == ShutdownState.RUNNING

    def test_get_shutdown_manager_returns_same(self):
        """Test get returns same instance."""
        manager1 = get_shutdown_manager()
        manager2 = get_shutdown_manager()

        assert manager1 is manager2

    def test_create_shutdown_manager(self):
        """Test create replaces global manager."""
        config = ShutdownConfig(timeout_seconds=60.0)

        manager1 = create_shutdown_manager(config)
        manager2 = get_shutdown_manager()

        assert manager1 is manager2
        assert manager1.config.timeout_seconds == 60.0

    def test_reset_shutdown_manager(self):
        """Test reset clears global manager."""
        manager1 = get_shutdown_manager()
        reset_shutdown_manager()
        manager2 = get_shutdown_manager()

        assert manager1 is not manager2


class TestShutdownIntegration:
    """Integration tests for shutdown."""

    @pytest.mark.asyncio
    async def test_full_shutdown_flow(self):
        """Test complete shutdown flow."""
        config = ShutdownConfig(
            timeout_seconds=5.0,
            grace_period_seconds=0.1,
            handle_sigterm=False,
            handle_sigint=False,
        )
        manager = ShutdownManager(config)

        # Track execution order
        execution_order = []

        async def hook1():
            execution_order.append("hook1")

        async def hook2():
            execution_order.append("hook2")

        def sync_hook():
            execution_order.append("sync")

        manager.register_hook("hook1", hook1, priority=10)
        manager.register_hook("sync", sync_hook, priority=20)
        manager.register_hook("hook2", hook2, priority=30)

        # Simulate in-flight request
        await manager.request_tracker.start_request()

        async def complete_request():
            await asyncio.sleep(0.2)
            await manager.request_tracker.end_request()

        asyncio.create_task(complete_request())

        # Initiate shutdown
        result = await manager.initiate_shutdown("Integration test")

        # Verify
        assert result.state == ShutdownState.STOPPED
        assert result.requests_drained == 1
        assert len(result.hooks_executed) == 3
        assert execution_order == ["hook1", "sync", "hook2"]
        assert result.duration_seconds > 0.1  # Grace period
