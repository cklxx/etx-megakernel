"""Supported ETX Python API (also re-exported from :mod:`etx`).

``__all__`` is the complete supported surface:

* ``plan`` verifies and schedules a graph.
* ``compile`` emits a plan's device sources and host metadata.
* ``load_model`` loads a validated model contract.
* ``run`` decodes tokens with an initialized runtime and descriptor.
* ``serve`` processes an iterable of descriptors synchronously.

Device compilation, weight allocation, callbacks and request transport remain
caller-owned. Importing the API does not load a GPU runtime library.
"""
from ._entrypoints import compile, plan, load_model
from etx.runtime.execution import run
from etx.serve import serve

__all__ = ['compile', 'plan', 'run', 'load_model', 'serve']
