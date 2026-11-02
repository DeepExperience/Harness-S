"""Restore mypy's legacy traceback-formatting hook on newer pytest versions.

Only traceback presentation is adapted; test execution and outcomes are intact.
Loaded explicitly for mypy task images, never as a global pytest plugin.
"""
from _pytest.nodes import Node


def _prunetraceback(self, excinfo):
    excinfo.traceback = self._traceback_filter(excinfo)


def pytest_configure(config):
    if not hasattr(Node, "_prunetraceback"):
        Node._prunetraceback = _prunetraceback
