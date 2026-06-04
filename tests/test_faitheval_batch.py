"""Batched FaithEval generation must not change results vs the batch-1 path.

The risk in batching greedy decode is that left-padding / prompt-stripping is wrong,
or that the chunked loop drops or reorders rows. These tests pin both with fakes so
they run without a GPU, model, or network.
"""

import torch

from src import faitheval_eval as fe


class _Result:
	def __init__(self, label):
		self.label = label
		self.method = "rule"
		self.reason = ""


def _fake_rows(n):
	return [{"qid": f"q{i}", "context": f"ctx{i}", "question": f"que{i}"} for i in range(n)]


def _patch_common(monkeypatch, rows):
	monkeypatch.setattr(
		fe, "load_config",
		lambda: {"faitheval": {"hf_dataset": "fake", "prompt_template": "{context}|{question}"}},
	)
	monkeypatch.setattr(fe, "load_dataset", lambda name, split=None: rows)
	# deterministic "generation": output is a pure function of the prompt, so the
	# batched path (defined here as exactly per-prompt) and batch-1 path must agree
	monkeypatch.setattr(
		fe, "_greedy_generate",
		lambda model, tok, prompt, pre_forward_hook=None: f"OUT[{prompt}]",
	)
	monkeypatch.setattr(
		fe, "_greedy_generate_batch",
		lambda model, tok, prompts, pre_forward_hook=None: [f"OUT[{p}]" for p in prompts],
	)
	monkeypatch.setattr(
		fe, "classify",
		lambda output, q, c, force_judge=False: _Result("refuses" if "que1|" in output else "fabricates"),
	)


def test_batch_matches_single(monkeypatch):
	rows = _fake_rows(20)
	_patch_common(monkeypatch, rows)

	df1 = fe.run_eval(None, None, batch_size=1)
	df8 = fe.run_eval(None, None, batch_size=8)

	expected_qids = [r["qid"] for r in rows]
	assert list(df1["qid"]) == expected_qids
	assert list(df8["qid"]) == expected_qids  # batching preserves order
	assert list(df1["output"]) == list(df8["output"])
	assert list(df1["label"]) == list(df8["label"])


def test_batch_size_not_dividing_dataset_keeps_all_rows(monkeypatch):
	rows = _fake_rows(20)
	_patch_common(monkeypatch, rows)

	# 20 % 7 != 0 — the final short chunk must not be dropped
	df = fe.run_eval(None, None, batch_size=7)
	assert len(df) == 20
	assert list(df["qid"]) == [r["qid"] for r in rows]


class _FakeBatch(dict):
	def to(self, device):
		return self


class _FakeTok:
	"""Minimal stand-in: records padding_side seen during tokenization, mimics HF's
	pad_token setter (setting pad_token to eos also sets pad_token_id)."""

	def __init__(self):
		self.padding_side = "right"
		self.eos_token = "<eos>"
		self.eos_token_id = 7
		self._pad_token = None
		self.pad_token_id = None
		self.side_during_call = None

	@property
	def pad_token(self):
		return self._pad_token

	@pad_token.setter
	def pad_token(self, value):
		self._pad_token = value
		if value == self.eos_token:
			self.pad_token_id = self.eos_token_id

	def __call__(self, prompts, return_tensors="pt", padding=True):
		self.side_during_call = self.padding_side
		# two rows left-padded to width 3 (row 1 has one pad on the left)
		ids = torch.tensor([[0, 1, 2], [7, 0, 3]])
		mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
		return _FakeBatch({"input_ids": ids, "attention_mask": mask})

	def decode(self, row, skip_special_tokens=True):
		return "+".join(str(int(t)) for t in row)


class _FakeModel:
	device = "cpu"
	dtype = torch.float32

	def generate(self, input_ids=None, attention_mask=None, **kwargs):
		gen = torch.tensor([[10, 11], [12, 13]])  # 2 new tokens per row
		return torch.cat([input_ids, gen], dim=1)


def test_batch_generate_left_pads_strips_and_restores():
	tok = _FakeTok()
	model = _FakeModel()

	out = fe._greedy_generate_batch(model, tok, ["a", "bb"])

	assert tok.side_during_call == "left"  # flipped to left for decoder-only generation
	assert tok.padding_side == "right"  # original side restored afterward
	assert tok.pad_token_id == 7  # pad set from eos when missing
	assert out == ["10+11", "12+13"]  # shared prompt width (3) sliced off, per-row decode


class _TemplateTok:
	"""Records whether the chat template or the raw tokenizer path was taken."""

	def __init__(self):
		self.used_chat_template = False
		self.used_raw = False
		self.eos_token_id = 0

	def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict, **kw):
		self.used_chat_template = True
		# the user content must be wrapped as a single user turn
		assert messages == [{"role": "user", "content": "hello"}]
		assert add_generation_prompt is True
		return _FakeBatch({"input_ids": torch.tensor([[1, 2]])})

	def __call__(self, prompt, return_tensors="pt"):
		self.used_raw = True
		return _FakeBatch({"input_ids": torch.tensor([[1, 2]])})

	def decode(self, row, skip_special_tokens=True):
		return "+".join(str(int(t)) for t in row)


class _OneStepModel:
	device = "cpu"
	dtype = torch.float32

	def generate(self, input_ids=None, max_new_tokens=None, **kwargs):
		return torch.cat([input_ids, torch.tensor([[9]])], dim=1)


def test_use_chat_template_routes_through_apply_chat_template():
	tok = _TemplateTok()
	fe._greedy_generate(_OneStepModel(), tok, "hello", use_chat_template=True)
	assert tok.used_chat_template and not tok.used_raw


def test_default_skips_chat_template():
	tok = _TemplateTok()
	fe._greedy_generate(_OneStepModel(), tok, "hello")  # default use_chat_template=False
	assert tok.used_raw and not tok.used_chat_template
