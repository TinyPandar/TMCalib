"""Composition root: the only place that chooses concrete implementations."""

from typing import Optional

from tmcalib.adapters.legacy import LegacyHardwareAdapter
from tmcalib.events import EventBus
from tmcalib.profiles import ProfileSpec, get_profile
from tmcalib.workflow import CalibrationWorkflow, RuntimeServices


def build_workflow(
    profile_name: str,
    channel: Optional[str] = None,
    events: Optional[EventBus] = None,
) -> CalibrationWorkflow:
    """Manually inject one lazily loaded hardware adapter into all ports."""
    profile: ProfileSpec = get_profile(profile_name, channel)
    adapter = LegacyHardwareAdapter(profile)
    services = RuntimeServices(
        camera=adapter,
        measurement=adapter,
        reconstruction=adapter,
        focus=adapter,
        lifecycle=adapter,
        cancellation=adapter,
    )
    return CalibrationWorkflow(profile=profile, services=services, events=events)
