"""
Batch logger for tracking fetch attempts.

Buffers fetch attempts in memory and flushes to SQLite in batches
for efficiency. Each fetch attempt (found, not_found, error, skipped)
is logged for completeness verification and gap detection.
"""

import atexit
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import MCFDatabase

logger = logging.getLogger(__name__)


class BatchLogger:
    """
    Buffers fetch attempts and flushes to database in batches.

    This reduces SQLite write overhead by batching inserts. The buffer
    is flushed automatically when it reaches batch_size, and also on
    cleanup via atexit.

    Trade-off: Max batch_size IDs could be lost on crash, but gap
    analysis can detect and retry missing IDs.

    Example:
        batch_logger = BatchLogger(db, batch_size=50)
        batch_logger.log(2023, 1, 'found')
        batch_logger.log(2023, 2, 'not_found')
        # ... automatically flushes every 50 attempts
        batch_logger.flush()  # Force final flush
    """

    def __init__(self, db: "MCFDatabase", batch_size: int = 50):
        """
        Initialize the batch logger.

        Args:
            db: MCFDatabase instance for writing attempts
            batch_size: Number of attempts to buffer before flushing
        """
        self.db = db
        self.batch_size = batch_size
        self._buffer: list[dict] = []
        self._total_logged = 0

        # Register cleanup handler to flush on exit
        atexit.register(self._cleanup)

    def log(
        self,
        year: int,
        sequence: int,
        result: str,
        error_message: str | None = None,
    ) -> None:
        """
        Add a fetch attempt to the buffer.

        Args:
            year: Year of the job ID
            sequence: Sequence number of the job ID
            result: Result type - 'found', 'not_found', 'error', or 'skipped'
            error_message: Optional error message for 'error' results
        """
        self._buffer.append({
            "year": year,
            "sequence": sequence,
            "result": result,
            "error_message": error_message,
        })

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """
        Write buffered attempts to database.

        Returns:
            Number of attempts written
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)
        try:
            self.db.batch_insert_attempts(self._buffer)
            self._total_logged += count
            logger.debug(f"Flushed {count} fetch attempts to database")
        except Exception as e:
            logger.error(f"Failed to flush {count} attempts: {e}")
            raise
        finally:
            self._buffer.clear()

        return count

    def _cleanup(self) -> None:
        """Cleanup handler - flush remaining buffer on exit."""
        if self._buffer:
            try:
                self.flush()
            except Exception as e:
                logger.error(f"Failed to flush buffer on cleanup: {e}")

    @property
    def pending_count(self) -> int:
        """Number of attempts pending in buffer."""
        return len(self._buffer)

    @property
    def total_logged(self) -> int:
        """Total attempts logged (flushed to database)."""
        return self._total_logged

    def __del__(self):
        """Destructor - attempt to flush on garbage collection."""
        if self._buffer:
            try:
                self.flush()
            except Exception:
                pass  # Suppress errors during garbage collection
