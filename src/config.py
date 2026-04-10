"""
Configuration dataclasses for the experiment pipeline.
Loaded from YAML files in configs/.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "EleutherAI/pythia-2.8b"
    transformer_lens_name: str = "pythia-2.8b"
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class SAEConfig:
    release: str = "pythia-70m-deduped-res-sm"
    sae_id: Optional[str] = None
    expansion_factor: int = 8
    
    # NEW: Local SAE checkpoint support
    local_sae_base_path: Optional[str] = None  # Base directory for local checkpoints
    local_sae_path_template: Optional[str] = None  # Template: "{base}/{model}/layer_{layer}/sae_weights.pt"


@dataclass
class LayersConfig:
    shallow: List[int] = field(default_factory=lambda: [2])
    deep: List[int] = field(default_factory=lambda: [12, 13, 14, 15, 16])
    intervention_layers: List[int] = field(default_factory=lambda: [2, 12, 14, 16])

    @property
    def all_layers(self) -> List[int]:
        return sorted(set(self.shallow + self.deep))


@dataclass
class PerformanceConfig:
    activation_cache_batch_size: int = 8
    sae_encode_batch_size: int = 512
    intervention_batch_size: int = 32


@dataclass
class DataConfig:
    gsm8k_split: str = "test"
    triviaqa_split: str = "validation"
    num_samples: int = 200
    max_seq_len: int = 512
    cache_dir: str = "data/activations"
    use_triviaqa: bool = True


@dataclass
class PromptsConfig:
    cot_prefix: str = "Let's solve this step by step.\n"
    no_cot_prefix: str = "Answer directly.\n"
    cot_suffix: str = "\nTherefore, the answer is"
    no_cot_suffix: str = "\nThe answer is"


@dataclass
class ContrastiveConfig:
    activation_diff_threshold: float = 0.5
    top_percentile: float = 95.0
    intersection_method: str = "strict"


@dataclass
class InterventionConfig:
    k_values: List[int] = field(default_factory=lambda: [10, 50, 100, 200])
    ablation_value: float = 0.0
    amplification_scales: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    num_random_trials: int = 5
    max_new_tokens: int = 256


@dataclass
class EvaluationConfig:
    answer_regex: str = r"#### (\-?[\d,]+\.?\d*)"
    perplexity_model: Optional[str] = None
    perplexity_stride: int = 256


@dataclass
class SweepModelEntry:
    name: str
    tl_name: str
    sae_release: str
    layers: List[int]


@dataclass
class SweepConfig:
    models: List[SweepModelEntry] = field(default_factory=list)


@dataclass
class OutputConfig:
    results_dir: str = "outputs"
    save_activations: bool = True
    save_generated_text: bool = True
    log_level: str = "INFO"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    layers: LayersConfig = field(default_factory=LayersConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "sae" in raw:
            cfg.sae = SAEConfig(**raw["sae"])
        if "layers" in raw:
            cfg.layers = LayersConfig(**raw["layers"])
        if "data" in raw:
            cfg.data = DataConfig(**raw["data"])
        if "prompts" in raw:
            cfg.prompts = PromptsConfig(**raw["prompts"])
        if "contrastive" in raw:
            cfg.contrastive = ContrastiveConfig(**raw["contrastive"])
        if "interventions" in raw:
            cfg.interventions = InterventionConfig(**raw["interventions"])
        if "evaluation" in raw:
            cfg.evaluation = EvaluationConfig(**raw["evaluation"])
        if "sweep" in raw:
            models = [SweepModelEntry(**m) for m in raw["sweep"].get("models", [])]
            cfg.sweep = SweepConfig(models=models)
        if "output" in raw:
            cfg.output = OutputConfig(**raw["output"])
        if "performance" in raw:
            cfg.performance = PerformanceConfig(**raw["performance"])
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a nested dict (for logging / saving)."""
        import dataclasses
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        return _to_dict(self)
