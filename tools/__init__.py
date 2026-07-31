"""Search Engine Agent Tools Package."""

# Import the compatibility boundary before callers import public security helpers.
# The boundary patches the implementation module in place, matching the existing
# integrity/rag compatibility-layer architecture.
from tools import security_boundary as _security_boundary  # noqa: F401,E402
