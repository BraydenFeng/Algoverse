"""M5 mediation: rate conventions, the clamp-only hook arithmetic, and the
screen's go/no-go thresholds. The arms need a GPU; these pin the logic that
decides what the numbers mean."""

import types

import pandas as pd
import torch

from src.m5_mediation import _clamp_only_hook, _rates, _steering_tensor


class _IdentityLayer(torch.nn.Module):
	def forward(self, x):
		return x


class _FakeModel:
	def __init__(self, layer):
		self.model = types.SimpleNamespace(layers=[layer])
		self.device = "cpu"
		self.dtype = torch.float32


class _FakeSAE:
	"""Identity encoder so encode(h)[j] == h[j]."""

	def __init__(self, w_dec):
		self._p = torch.nn.Parameter(torch.zeros(1))
		self.W_dec = w_dec

	def parameters(self):
		return iter([self._p])

	def encode(self, x):
		return x


def test_rates_ignore_empty_generations():
	df = pd.DataFrame({
		"output": ["a", "", "  ", "d"],
		"label": ["refuses", "off_topic", "off_topic", "fabricates"],
	})
	r = _rates(df)
	assert r["n_total"] == 4 and r["n_nonempty"] == 2
	assert abs(r["refuse"] - 0.5) < 1e-9
	assert abs(r["fabricate"] - 0.5) < 1e-9


def test_rates_survive_nan_output():
	df = pd.DataFrame({"output": [None, "x"], "label": ["off_topic", "refuses"]})
	assert _rates(df)["n_nonempty"] == 1


def test_steering_tensor_magnitude_is_alpha_times_norm_scale():
	model = _FakeModel(_IdentityLayer())
	v = torch.tensor([3.0, 4.0]).numpy()  # norm 5 -> unit-normalized internally
	t = _steering_tensor(v, alpha=0.5, norm_scale=100.0, model=model)
	assert abs(float(t.norm()) - 50.0) < 1e-4


def test_clamp_only_hook_swaps_latent_contribution():
	layer = _IdentityLayer()
	w = torch.zeros(4, 4)
	w[2, 0] = 1.0  # latent 2 decodes onto dim 0
	factory = _clamp_only_hook(_FakeSAE(w), latent_idx=2, layer=0, tok_idx=0,
							   clamp_value=10.0, capture={})
	handle = factory(_FakeModel(layer))
	out = layer(torch.tensor([[[1.0, 1, 1, 1]]]))
	handle.remove()
	# a_2 = 1.0 -> delta = (10 - 1) on dim 0
	assert torch.allclose(out, torch.tensor([[[10.0, 1.0, 1.0, 1.0]]]), atol=1e-5)


def test_clamp_only_hook_noops_when_token_out_of_range():
	layer = _IdentityLayer()
	factory = _clamp_only_hook(_FakeSAE(torch.zeros(4, 4)), latent_idx=2, layer=0,
							   tok_idx=99, clamp_value=10.0, capture={})
	handle = factory(_FakeModel(layer))
	x = torch.tensor([[[1.0, 1, 1, 1]]])
	out = layer(x)
	handle.remove()
	assert torch.allclose(out, x)
