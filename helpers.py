"""Minimal stand-ins for the helpers imported by main.py.

The original helpers.py was never committed, so main.py could not be imported.
These no-op versions make it runnable; replace them with the originals if you
still have them.
"""


class EmergencyResponse:
    """Hook for forcing an action before the agent chooses one."""

    def check_emergency(self, self_state):
        return None


class ResourceBuffer:
    """Placeholder; main.py only attaches it to the agent."""
