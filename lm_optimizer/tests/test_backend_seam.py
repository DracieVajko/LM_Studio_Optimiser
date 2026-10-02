"""Backend seam conformance tests (Task 1)."""

from lm_optimizer.backends.base import assert_conforms
from lm_optimizer.services.lm_studio import LMStudioClient


def test_lm_studio_conforms_to_backend_client():
    assert_conforms(LMStudioClient())
