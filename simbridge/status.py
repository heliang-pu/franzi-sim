"""V1R1C9 business status codes and the bridge's error types.

Values copied from the vendor client (core/types/common.py SdkStatus, plus -8
for an active emergency stop); the vendor files themselves are not imported.
"""

from enum import IntEnum


class Sdk(IntEnum):
    OK = 0
    PARAM_INVALID = -3
    UNSUPPORTED = -4
    NOT_READY = -5
    # Vendor text: Cartesian target unreachable (IK, joint limits, ...).
    UNREACHABLE = -6
    ESTOP_ACTIVE = -8
    UNIT_FAULT = -9
    MODE_CONFLICT = -10


class Unsupported(Exception):
    """The method or option exists in the SDK but this backend cannot provide it.

    Sent as a protocol error so the stock client always raises, whatever its
    ``raise_on_error`` setting; a silent empty default would be a false success.
    """


class InvalidRequest(ValueError):
    """Malformed arguments (wrong type or shape) before any business check."""


class Rejected(Exception):
    """A business-level refusal, answered with an SDK status code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class PhysicsFault(RuntimeError):
    """A per-step physics audit failed (penetration, joint limit, solver)."""

    def __init__(self, kind, message, **details):
        super().__init__(message)
        self.kind = kind
        self.details = details
