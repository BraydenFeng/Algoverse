"""M4 mediation: Imai decomposition math and the SAE-latent intervention hooks.

The hooks are the load-bearing, easy-to-get-wrong part (capture the right token,
swap only latent j's contribution, steer-then-clamp in one pass). These pin the
residual arithmetic with a fake identity SAE so they run without a model/GPU.
"""

import types

import pytest
import torch

from src.m4_mediation import (
	compute_mediation,
	make_mediator_capture_hook,
	make_mediator_clamp_hook,
	make_rescue_hook,
	_entity_position,
	_last_prompt_token_idx,
)


class _IdentityLayer(torch.nn.Module):
	def forward(self, x):
		return x


class _FakeModel:
	def __init__(self, layer):
		self.model = types.SimpleNamespace(layers=[layer])
		self.device = "cpu"
		self.dtype = torch.float32


class _FakeSAE:
	"""Identity encoder (d_sae == d_model) so encode(h)[j] == h[j] — lets tests
	assert exact latent values and decoder swaps."""

	def __init__(self, w_dec=None, d=4):
		self._p = torch.nn.Parameter(torch.zeros(1))
		self.W_dec = w_dec if w_dec is not None else torch.zeros(d, d)

	def parameters(self):
		return iter([self._p])

	def encode(self, x):
		return x


def _unit_row(idx, d=4):
	"""Decoder matrix whose row `idx` is e_0 = [1,0,...] and the rest zero."""
	w = torch.zeros(d, d)
	w[idx, 0] = 1.0
	return w


def test_compute_mediation_decomposition():
	rates = {
		"A": {"refuse": 0.30, "fabricate": 0.60, "off_topic": 0.10},
		"B": {"refuse": 0.10, "fabricate": 0.80, "off_topic": 0.10},
		"C": {"refuse": 0.25, "fabricate": 0.65, "off_topic": 0.10},
	}
	res = compute_mediation(rates, {"A": 100, "B": 100, "C": 100}, {"A": 1.0, "B": 0.2})
	assert abs(res.te - (0.10 - 0.30)) < 1e-9    # B - A
	assert abs(res.acme - (0.10 - 0.25)) < 1e-9  # B - C
	assert abs(res.ade - (0.25 - 0.30)) < 1e-9   # C - A
	assert abs(res.te_check - res.te) < 1e-9     # ACME + ADE closes to TE


def test_last_prompt_token_idx():
	class _Tok:
		def __call__(self, prompt, add_special_tokens=True):
			# pretend: one token per word + 1 special prefix
			return types.SimpleNamespace(input_ids=list(range(len(prompt.split()) + 1)))

	assert _last_prompt_token_idx(_Tok(), "alpha beta gamma") == 3  # 3 words + 1 special - 1


class _ForwardModel:
	"""Callable model whose __call__ runs the captured layer so a forward hook
	fires — minimal stand-in for `_entity_position`'s single forward pass."""

	def __init__(self, layer, resid):
		self.model = types.SimpleNamespace(layers=[layer])
		self._resid = resid

	def __call__(self, **enc):
		return self.model.layers[0](self._resid)


def test_entity_position_returns_argmax_firing_token():
	# latent 2 fires hardest at token 1 (value 9) -> entity position is 1, not last
	resid = torch.tensor([[[0.0, 0, 0, 0], [1.0, 2, 9, 4], [0.0, 0, 1, 0]]])
	model = _ForwardModel(_IdentityLayer(), resid)
	enc = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
	assert _entity_position(model, _FakeSAE(), enc, layer=0, latent_idx=2) == 1


def test_entity_position_falls_back_to_last_token_when_latent_silent():
	# latent 2 never fires (all zero) -> fall back to last prompt token (T-1 = 2)
	resid = torch.tensor([[[0.0, 0, 0, 0], [1.0, 2, 0, 4], [0.0, 0, 0, 0]]])
	model = _ForwardModel(_IdentityLayer(), resid)
	enc = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
	assert _entity_position(model, _FakeSAE(), enc, layer=0, latent_idx=2) == 2


def test_capture_hook_reads_latent_at_target_token():
	layer = _IdentityLayer()
	model = _FakeModel(layer)
	cap = {}
	factory = make_mediator_capture_hook(_FakeSAE(), latent_idx=2, layer=0, target_token_idx=1, capture=cap)
	handle = factory(model)
	# (B=1, T=2, d=4); target token 1 = [1,2,3,4], identity-encoded -> latent 2 == 3.0
	layer(torch.tensor([[[0.0, 0, 0, 0], [1.0, 2, 3, 4]]]))
	handle.remove()
	assert abs(cap["value"] - 3.0) < 1e-6


def test_rescue_hook_steers_then_clamps():
	layer = _IdentityLayer()
	model = _FakeModel(layer)
	sae = _FakeSAE(w_dec=_unit_row(2))
	steer = torch.tensor([0.5, 0.5, 0.5, 0.5])
	factory = make_rescue_hook(
		sae, latent_idx=2, layer=0, target_token_idx=0,
		clamp_value=10.0, steering_vector_on_device=steer,
	)
	handle = factory(model)
	out = layer(torch.tensor([[[1.0, 1, 1, 1]]]))
	handle.remove()
	# steered residual = 1.5 everywhere -> a_2 = 1.5 -> delta = (10 - 1.5)*e_0 = 8.5 on dim 0
	assert torch.allclose(out, torch.tensor([[[10.0, 1.5, 1.5, 1.5]]]), atol=1e-5)


def test_clamp_hook_swaps_latent_contribution_without_steering():
	layer = _IdentityLayer()
	model = _FakeModel(layer)
	sae = _FakeSAE(w_dec=_unit_row(2))
	factory = make_mediator_clamp_hook(sae, latent_idx=2, layer=0, target_token_idx=0, clamp_value=10.0)
	handle = factory(model)
	out = layer(torch.tensor([[[1.0, 1, 1, 1]]]))
	handle.remove()
	# no steering: a_2 = 1.0 -> delta = (10 - 1)*e_0 = 9 on dim 0
	assert torch.allclose(out, torch.tensor([[[10.0, 1.0, 1.0, 1.0]]]), atol=1e-5)


def test_clamp_hook_rejects_inline_steering_factory():
	# the half-built steering path must fail loudly, not silently no-op
	factory = make_mediator_clamp_hook(
		_FakeSAE(), latent_idx=0, layer=0, target_token_idx=0,
		clamp_value=0.0, desperation_steering_factory=lambda m: None,
	)
	with pytest.raises(NotImplementedError):
		factory(_FakeModel(_IdentityLayer()))
