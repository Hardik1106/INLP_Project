# Example: Run a single direct-answer prompt from GSM8K

from src.data_pipeline import build_prompt_sets
from src.cache_activations import load_hooked_model
from src.config import ExperimentConfig

# Load config and model as usual
config = ExperimentConfig.from_yaml("configs/default.yaml")
model = load_hooked_model(
    config.model.transformer_lens_name,
    config.model.device,
    config.model.dtype,
)

# Build prompt sets
prompt_sets = build_prompt_sets(config.data, config.prompts)

# Pick a single GSM8K No-CoT sample
sample = prompt_sets["gsm8k_no_cot"][0]  # First sample

# Generate output
tokens = model.to_tokens(sample.prompt_text, prepend_bos=True)
output_tokens = model.generate(
    tokens,
    max_new_tokens=10,
    do_sample=False,
    temperature=1.0,
    stop_at_eos=True,
    eos_token_id=model.tokenizer.eos_token_id,
)
completion = model.to_string(output_tokens[0][tokens.shape[1]:])
print("Prompt:", sample.prompt_text)
print("Model output:", completion)
print("Ground truth:", sample.ground_truth)