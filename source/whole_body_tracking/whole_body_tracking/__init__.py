"""
Python module serving as a project/extension template.
"""

import os as _os

# Register Gym environments. Skipped when ``WHOLE_BODY_TRACKING_NO_TASKS=1`` is set —
# used by the offline ``synth`` tooling so it doesn't pull in Isaac just to write an
# npz. Default (unset) preserves the prior import-on-load behaviour.
if not _os.environ.get("WHOLE_BODY_TRACKING_NO_TASKS"):
    from .tasks import *  # noqa: F401,F403
