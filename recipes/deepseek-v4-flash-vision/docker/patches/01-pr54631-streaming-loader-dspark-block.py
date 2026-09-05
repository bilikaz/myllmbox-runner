"""vLLM PR #54631 (lucamotz) — port of the two runtime fixes, as anchor patches.

(a) DeepseekV4ForConditionalGeneration.load_weights() on the image sorts the fully-mapped checkpoint
    iterator → every tensor resident in host RAM before TP sharding (157G on a 119G UMA box). Keep the
    iterator one-pass; the fused MegaMoE/MHC finalisation moves to the model-level
    process_weights_after_loading() hook (called by the loader once the stream is consumed), and the
    text model's own load_weights must no longer finalise early (it runs per contiguous group now).
(b) DSpark draft config: the Vision checkpoint declares num_nextn_predict_layers=3 but was trained with
    dspark_block_size=5. Derive n_predict from dspark_block_size (fail closed on a bad value) so k=5
    is legal and the draft emits the trained block width.
"""
import pathlib, vllm

root = pathlib.Path(vllm.__file__).parent

# ---- (a1) vl_model.py: one-pass loader + finalize in the post-load hook ------------------------------
p = root / "models/deepseek_v4/nvidia/vl_model.py"
s = p.read_text()
assert "MBX: PR #54631" not in s, "already patched"
old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF names into this wrapper's namespace up front and sort, so
        # the "language_model." group reaches the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model).
        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(mapped)
        # The child's load_weights already ran its post-load finalization.
        self._weights_finalized = True
        return loaded_params

    def process_weights_after_loading(self) -> None:
        # Model-level post-load hook (called by the loader after any load
        # format). Under DummyModelLoader the child's load_weights — and
        # hence its finalize step — is bypassed, so run it here instead.
        if getattr(self, "_weights_finalized", False):
            return
        self.language_model.process_weights_after_loading()
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # MBX: PR #54631 — one-pass streaming load. No sorted(): that materialised the
        # whole checkpoint in host RAM before TP sharding. Fused MegaMoE/MHC finalisation
        # happens in process_weights_after_loading() once the full stream is consumed.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()
'''
assert s.count(old) == 1, "vl_model load_weights anchor not found/unique — upstream changed"
s = s.replace(old, new, 1)
old2 = '''            # The MTP/DSpark draft heads are not supported for the vision
            # variant; drop their weights.
            "mtp.": None,'''
new2 = '''            # Draft heads are loaded independently by the MTP/DSpark model;
            # the target wrapper must not consume their weights.
            "mtp.": None,'''
assert s.count(old2) == 1, "vl_model mapper comment anchor not found/unique"
s = s.replace(old2, new2, 1)
p.write_text(s)
print("vl_model.py: streaming load_weights + post-load finalize (PR #54631)")

# ---- (a2) model.py: text model no longer finalises inside load_weights --------------------------------
p = root / "models/deepseek_v4/nvidia/model.py"
s = p.read_text()
old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.process_weights_after_loading()
        return loaded_params
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        # MBX: PR #54631 — finalisation deferred to process_weights_after_loading().
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
assert s.count(old) == 1, "model.py load_weights anchor not found/unique — upstream changed"
p.write_text(s.replace(old, new, 1))
print("model.py: load_weights no longer finalises early (PR #54631)")

# ---- (b) speculative.py: n_predict from dspark_block_size ------------------------------------------------
p = root / "config/speculative.py"
s = p.read_text()
assert "_normalize_deepseek_v4_dspark_hf_config" not in s, "already patched"
old = '''    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:'''
new = '''    @staticmethod
    def _normalize_deepseek_v4_dspark_hf_config(
        hf_config: PretrainedConfig,
    ) -> None:
        # MBX: PR #54631 — DSpark emits dspark_block_size tokens per draft call
        # (5 on Vision-Exp); num_nextn_predict_layers (3) is the MTP depth, not
        # the block width. Fail closed on a missing/bogus value.
        block_size = getattr(hf_config, "dspark_block_size", None)
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size <= 0
        ):
            raise ValueError(
                "DeepSeek-V4 DSpark requires a positive integer "
                "dspark_block_size in the checkpoint config."
            )
        hf_config.model_type = "deepseek_v4"
        hf_config.architectures = ["DSparkDraftModel"]
        hf_config.n_predict = block_size

    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:'''
assert s.count(old) == 1, "speculative.py _is_custom_proposer_path anchor not found/unique"
s = s.replace(old, new, 1)
old2 = '''                    # DeepSeek-V4 DSpark reuses the full DeepSeek-V4 config
                    # and its weights ship in the target checkpoint.
                    self.draft_model_config.hf_config.model_type = "deepseek_v4"
                    self.draft_model_config.hf_config.architectures = [
                        "DSparkDraftModel"
                    ]
'''
new2 = '''                    # DeepSeek-V4 DSpark reuses the full DeepSeek-V4 config
                    # and its weights ship in the target checkpoint.
                    SpeculativeConfig._normalize_deepseek_v4_dspark_hf_config(
                        self.draft_model_config.hf_config
                    )
'''
assert s.count(old2) == 1, "speculative.py DSparkDraftModel anchor not found/unique"
s = s.replace(old2, new2, 1)
assert "PretrainedConfig" in s, "PretrainedConfig not imported in speculative.py — check import"
p.write_text(s)
print("speculative.py: DeepSeek-V4 DSpark n_predict := dspark_block_size (PR #54631)")
