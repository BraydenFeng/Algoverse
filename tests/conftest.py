"""Stub heavy optional deps so the pure-logic tests run on machines without the
full model stack (e.g. the dev laptop). On the GPU box the real packages import
normally and these stubs are never installed."""

import sys
import types

try:
	import datasets  # noqa: F401
except ImportError:
	_stub = types.ModuleType("datasets")
	_stub.load_dataset = lambda *args, **kwargs: None
	sys.modules["datasets"] = _stub
