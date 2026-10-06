"""Evaluation module: QA dataset, retrieval evaluation, ragas quality eval."""

from .compat import install as _install_compat

# Must run before any `import ragas`. The shim is idempotent and cheap: it only
# stubs a module that langchain-community 0.4.x removed but ragas 0.4.x still
# re-exports at import time.
_install_compat()
