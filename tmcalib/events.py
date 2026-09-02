"""Framework-neutral workflow events used by Qt, CLI, and tests."""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional


class EventKind(str, Enum):
    STATE = "state"
    LOG = "log"
    PROGRESS = "progress"
    FRAME = "frame"
    RESULT = "result"
    ERROR = "error"


@dataclass(frozen=True)
class WorkflowEvent:
    kind: EventKind
    operation: str
    message: str = ""
    progress: Optional[float] = None
    payload: Any = None
    metadata: Optional[Mapping[str, Any]] = None


EventCallback = Callable[[WorkflowEvent], None]


class EventBus:
    """Small thread-safe publisher; subscribers own any UI-thread marshalling."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventKind, List[EventCallback]] = {
            kind: [] for kind in EventKind
        }
        self._lock = threading.RLock()

    def subscribe(self, kind: EventKind, callback: EventCallback) -> Callable[[], None]:
        with self._lock:
            self._subscribers[kind].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers[kind]
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def publish(self, event: WorkflowEvent) -> None:
        with self._lock:
            callbacks = tuple(self._subscribers[event.kind])
        for callback in callbacks:
            callback(event)
