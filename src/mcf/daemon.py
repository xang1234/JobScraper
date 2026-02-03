"""
Scraper daemon for long-running background operation.

Provides a simple daemon process manager with:
- Background process execution via fork
- Heartbeat-based wake detection (detects sleep/resume)
- PID file management
- Status monitoring

Note: This module uses Unix-specific features (os.fork, os.setsid)
and will only work on Unix-like systems (Linux, macOS).
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable, Any

if TYPE_CHECKING:
    from .database import MCFDatabase

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds
DEFAULT_WAKE_THRESHOLD = 300  # 5 minutes - gap indicates sleep/wake


class DaemonError(Exception):
    """Base exception for daemon errors."""
    pass


class DaemonAlreadyRunning(DaemonError):
    """Raised when trying to start a daemon that's already running."""
    pass


class DaemonNotRunning(DaemonError):
    """Raised when trying to stop a daemon that's not running."""
    pass


class ScraperDaemon:
    """
    Simple daemon process manager for long-running scrapes.

    Features:
    - Forks into background process
    - Maintains heartbeat for wake detection
    - PID file for process management
    - Database state tracking

    Example:
        daemon = ScraperDaemon(db)

        # Start in background
        daemon.start(scrape_year_func, year=2023)

        # Check status
        status = daemon.status()
        print(f"Running: {status['status'] == 'running'}")

        # Stop
        daemon.stop()
    """

    def __init__(
        self,
        db: "MCFDatabase",
        pidfile: str | Path = "data/.scraper.pid",
        logfile: str | Path = "data/scraper_daemon.log",
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        wake_threshold: int = DEFAULT_WAKE_THRESHOLD,
    ):
        """
        Initialize the daemon manager.

        Args:
            db: MCFDatabase for state tracking
            pidfile: Path to PID file
            logfile: Path to daemon log file
            heartbeat_interval: Seconds between heartbeats
            wake_threshold: Seconds gap that indicates sleep/wake
        """
        self.db = db
        self.pidfile = Path(pidfile)
        self.logfile = Path(logfile)
        self.heartbeat_interval = heartbeat_interval
        self.wake_threshold = wake_threshold

        # Ensure parent directories exist
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.logfile.parent.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        if not self.pidfile.exists():
            return False

        try:
            pid = int(self.pidfile.read_text().strip())
            # Check if process exists (signal 0 doesn't kill, just checks)
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            # PID file exists but process doesn't - stale PID file
            self.pidfile.unlink(missing_ok=True)
            return False

    def get_pid(self) -> int | None:
        """Get PID of running daemon, or None if not running."""
        if not self.pidfile.exists():
            return None
        try:
            return int(self.pidfile.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def start(
        self,
        scraper_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> int:
        """
        Fork and run scraper in background.

        Args:
            scraper_func: Async function to run (e.g., scraper.scrape_year)
            *args: Positional arguments for scraper_func
            **kwargs: Keyword arguments for scraper_func

        Returns:
            PID of the daemon process

        Raises:
            DaemonAlreadyRunning: If daemon is already running
        """
        if self.is_running():
            pid = self.get_pid()
            raise DaemonAlreadyRunning(f"Daemon already running with PID {pid}")

        # First fork
        try:
            pid = os.fork()
        except OSError as e:
            raise DaemonError(f"First fork failed: {e}")

        if pid > 0:
            # Parent process: wait briefly for child to start
            time.sleep(0.1)
            # Read PID from file (child writes it)
            child_pid = self.get_pid()
            if child_pid:
                return child_pid
            return pid

        # Child process: become session leader
        os.setsid()

        # Second fork (prevent acquiring terminal)
        try:
            pid = os.fork()
        except OSError as e:
            sys.exit(1)

        if pid > 0:
            # First child exits
            sys.exit(0)

        # Grandchild: actual daemon process
        self._run_daemon(scraper_func, *args, **kwargs)
        sys.exit(0)

    def _run_daemon(
        self,
        scraper_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> None:
        """Run the daemon process (called after forking)."""
        # Write PID file
        self.pidfile.write_text(str(os.getpid()))

        # Set up logging to file
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)

        # Redirect stdout/stderr to log file
        sys.stdout = open(self.logfile, 'a')
        sys.stderr = sys.stdout

        # Close inherited file descriptors
        os.close(0)  # stdin

        logger.info(f"Daemon started with PID {os.getpid()}")

        # Update database state
        self.db.update_daemon_state(os.getpid(), 'running')

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Run the scraper with heartbeat
        try:
            asyncio.run(self._run_with_heartbeat(scraper_func, *args, **kwargs))
        except Exception as e:
            logger.exception(f"Daemon error: {e}")
        finally:
            self._cleanup()

    async def _run_with_heartbeat(
        self,
        scraper_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> Any:
        """Run scraper function with concurrent heartbeat task."""
        last_beat = time.time()
        should_stop = asyncio.Event()

        async def heartbeat_loop():
            nonlocal last_beat
            while not should_stop.is_set():
                now = time.time()
                gap = now - last_beat

                if gap > self.wake_threshold:
                    logger.warning(
                        f"Wake detected: {gap:.0f}s gap (threshold: {self.wake_threshold}s). "
                        "Resuming scrape..."
                    )

                last_beat = now
                self.db.update_daemon_heartbeat()

                try:
                    await asyncio.wait_for(
                        should_stop.wait(),
                        timeout=self.heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    pass  # Expected - continue heartbeat loop

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_loop())

        try:
            # Run the scraper
            result = await scraper_func(*args, **kwargs)
            return result
        finally:
            # Stop heartbeat
            should_stop.set()
            await heartbeat_task

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Clean up daemon state."""
        try:
            self.db.update_daemon_state(os.getpid(), 'stopped')
        except Exception as e:
            logger.error(f"Failed to update daemon state: {e}")

        try:
            self.pidfile.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")

        logger.info("Daemon stopped")

    def stop(self, timeout: int = 10) -> bool:
        """
        Stop the running daemon.

        Args:
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if daemon was stopped, False if it wasn't running

        Raises:
            DaemonNotRunning: If no daemon is running
            DaemonError: If daemon couldn't be stopped
        """
        if not self.is_running():
            raise DaemonNotRunning("No daemon is running")

        pid = self.get_pid()
        if not pid:
            raise DaemonNotRunning("No daemon is running")

        logger.info(f"Stopping daemon (PID {pid})...")

        # Send SIGTERM for graceful shutdown
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process already gone
            self.pidfile.unlink(missing_ok=True)
            return True

        # Wait for process to exit
        for _ in range(timeout):
            time.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # Process exited
                self.pidfile.unlink(missing_ok=True)
                return True

        # Process didn't exit - force kill
        logger.warning(f"Daemon didn't exit gracefully, sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        except ProcessLookupError:
            pass

        self.pidfile.unlink(missing_ok=True)
        return True

    def status(self) -> dict:
        """
        Get daemon status.

        Returns:
            Dict with status information including:
            - running: bool
            - pid: int or None
            - database_state: dict from daemon_state table
        """
        running = self.is_running()
        pid = self.get_pid() if running else None
        db_state = self.db.get_daemon_state()

        return {
            "running": running,
            "pid": pid,
            "pidfile": str(self.pidfile),
            "logfile": str(self.logfile),
            **db_state,
        }
