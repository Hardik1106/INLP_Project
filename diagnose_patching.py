"""
Diagnostic script to find why all patching deltas are exactly 0.0.
Tests: SAE codes, hook firing, activation differences.
"""
import os, sys, pickle, torch
import torch.nn.functional as F

SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_coding")
sys.path.insert(0, SC)
from device_utils import get_device, get_model_dtype
from transformer_lens import HookedTransformer

device = str(get_device())
dtype = get_model_dtype()

# 1. Load model
print("="*60)
print("1. Loading model...")
model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-70m", device=device, torch_dtype=dtype,
    fold_ln=False, center_writing_weights=False,
)
tokenizer = model.tokenizer
print(f"   Model loaded on {device}, dtype={dtype}")

# 2. Load SAEs
print("\n2. Loading SAEs...")
nocot_pkl = os.path.join(SC, "output", "nocot_sae_r4", "dict.pkl")
cot_pkl = os.path.join(SC, "output", "cot_sae_r4", "dict.pkl")
with open(nocot_pkl, "rb") as f:
    dict_nc = pickle.load(f)
with open(cot_pkl, "rb") as f:
    dict_c = pickle.load(f)

layer = 2
rank = 2
sae_nc = dict_nc[layer][rank][0]
sae_c = dict_c[layer][rank][0]
sae_nc.to_device(device)
sae_c.to_device(device)
print(f"   SAE type: {type(sae_nc).__name__}")
print(f"   n_feats={sae_nc.n_feats}, activation_size={sae_nc.activation_size}")
print(f"   encoder dtype: {sae_nc.encoder.dtype}, device: {sae_nc.encoder.device}")

# 3. Load one chunk of activations
print("\n3. Loading activations (chunk 0)...")
nc_acts_dir = os.path.join(SC, "data", "nocot_acts")
c_acts_dir = os.path.join(SC, "data", "cot_acts")
acts_nc = torch.load(os.path.join(nc_acts_dir, "0.pt"), map_location="cpu").to(torch.float32)
acts_c = torch.load(os.path.join(c_acts_dir, "0.pt"), map_location="cpu").to(torch.float32)
ids_list = torch.load(os.path.join(nc_acts_dir, "input_ids_layer2_chunk0.pt"))
print(f"   acts_nc shape: {acts_nc.shape}")
print(f"   acts_c shape: {acts_c.shape}")
print(f"   ids_list type: {type(ids_list)}, shape: {ids_list.shape if hasattr(ids_list, 'shape') else 'N/A'}")

# 4. Test SAE encode/decode on first example
print("\n4. Testing SAE encode/decode...")
if isinstance(ids_list, torch.Tensor) and ids_list.dim() == 2:
    ids = ids_list[0]  # first example, shape (256,)
else:
    ids = ids_list[0]
    
seq_len = ids.shape[0]
idx_end = seq_len - 1  # last token position in flat activation tensor
print(f"   First example: seq_len={seq_len}, idx_end={idx_end}")

act_nc_last = acts_nc[idx_end:idx_end+1].to(device).to(sae_nc.encoder.dtype)
act_c_last = acts_c[idx_end:idx_end+1].to(device).to(sae_c.encoder.dtype)

print(f"   act_nc_last: shape={act_nc_last.shape}, norm={act_nc_last.norm().item():.4f}")
print(f"   act_c_last:  shape={act_c_last.shape}, norm={act_c_last.norm().item():.4f}")
print(f"   acts equal? {torch.allclose(act_nc_last, act_c_last)}")
print(f"   acts max diff: {(act_nc_last - act_c_last).abs().max().item():.6f}")

code_nc = sae_nc.encode(act_nc_last)
code_c = sae_c.encode(act_c_last)
print(f"\n   code_nc: shape={code_nc.shape}, nnz={(code_nc > 0).sum().item()}, max={code_nc.max().item():.4f}")
print(f"   code_c:  shape={code_c.shape}, nnz={(code_c > 0).sum().item()}, max={code_c.max().item():.4f}")

diff = (code_c - code_nc).abs().squeeze(0)
print(f"   code diff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}, nnz={diff.nonzero().shape[0]}")
topk = diff.topk(k=20)
print(f"   top-20 diffs: {topk.values.cpu().tolist()[:10]}")

# 5. Test patched activation
print("\n5. Testing patched activation...")
patched_code = code_nc.clone()
patched_code[0, topk.indices] = code_c[0, topk.indices]
patched_act = sae_nc.decode(patched_code).squeeze(0)

# What does original decode look like?
orig_decoded = sae_nc.decode(code_nc).squeeze(0)
print(f"   orig_decoded norm: {orig_decoded.norm().item():.4f}")
print(f"   patched_act norm:  {patched_act.norm().item():.4f}")
print(f"   patched == orig? {torch.allclose(patched_act, orig_decoded)}")
print(f"   patched - orig max diff: {(patched_act - orig_decoded).abs().max().item():.6f}")

# 6. Check if TransformerLens hooks actually fire
print("\n6. Testing TransformerLens hook mechanism...")
prompt_ids = ids.unsqueeze(0).to(device)

# Simple test: check if we can modify resid_pre
hook_fired = [False]
def test_hook(resid, hook):
    hook_fired[0] = True
    print(f"   HOOK FIRED! resid shape={resid.shape}, device={resid.device}")
    return resid

# Test with model.hooks() context manager
print("   Testing model.hooks() context manager...")
try:
    with model.hooks([(f"blocks.{layer}.hook_resid_pre", test_hook)]):
        _ = model(prompt_ids, return_type="logits")
    print(f"   Hook fired (context manager)? {hook_fired[0]}")
except Exception as e:
    print(f"   ERROR with model.hooks(): {e}")

# Test with fwd_hooks kwarg
hook_fired[0] = False
print("   Testing model.hooks(fwd_hooks=...) ...")
try:
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", test_hook)]):
        _ = model(prompt_ids, return_type="logits")
    print(f"   Hook fired (fwd_hooks kwarg)? {hook_fired[0]}")
except Exception as e:
    print(f"   ERROR with fwd_hooks kwarg: {e}")

# Test with run_with_hooks
hook_fired[0] = False
print("   Testing model.run_with_hooks()...")
try:
    _ = model.run_with_hooks(
        prompt_ids,
        return_type="logits",
        fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", test_hook)]
    )
    print(f"   Hook fired (run_with_hooks)? {hook_fired[0]}")
except Exception as e:
    print(f"   ERROR with run_with_hooks: {e}")

# 7. Test actual patching effect
print("\n7. Testing actual patching effect on logits...")
# Get baseline logits
with torch.no_grad():
    baseline_logits = model(prompt_ids, return_type="logits")
    baseline_last = baseline_logits[0, -1, :].clone()

# Get patched logits
def patch_hook(resid, hook):
    resid[:, -1, :] = patched_act.to(resid.device)
    return resid

try:
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)]):
        patched_logits = model(prompt_ids, return_type="logits")
        patched_last = patched_logits[0, -1, :].clone()
    
    logit_diff = (patched_last - baseline_last).abs()
    print(f"   Logit diff at last pos: max={logit_diff.max().item():.6f}, mean={logit_diff.mean().item():.6f}")
except Exception as e:
    print(f"   ERROR: {e}")

# Also test with run_with_hooks
try:
    patched_logits2 = model.run_with_hooks(
        prompt_ids,
        return_type="logits",
        fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)]
    )
    patched_last2 = patched_logits2[0, -1, :].clone()
    logit_diff2 = (patched_last2 - baseline_last).abs()
    print(f"   run_with_hooks logit diff: max={logit_diff2.max().item():.6f}, mean={logit_diff2.mean().item():.6f}")
except Exception as e:
    print(f"   ERROR: {e}")

# 8. Test the actual answer_logprob function
print("\n8. Testing answer_logprob function...")
from unicodedata import normalize
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="train", verification_mode="no_checks")
# Find the first matched question
decoded_prompt = tokenizer.decode(ids, skip_special_tokens=True)
print(f"   Decoded prompt (first 120 chars): {decoded_prompt[:120]}...")

# Try extracting question
try:
    q = decoded_prompt.split("Q:")[1].split("\nA:")[0]
except:
    q = decoded_prompt
q = normalize("NFKC", q.strip())
print(f"   Extracted question (first 80): {q[:80]}...")

# Find answer in dataset
lookup = {}
for ex in ds:
    q_txt = normalize("NFKC", ex["question"].strip())
    a = ex["answer"].split("####")[-1].strip()
    lookup[q_txt] = a

ans = lookup.get(q)
print(f"   Answer found? {ans is not None}")
if ans:
    print(f"   Answer: {ans}")
    
    def answer_logprob(mdl, pid, answer, hooks=None):
        ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        full_ids = torch.cat(
            [pid.squeeze(0), torch.tensor(ans_ids, device=pid.device)], 0
        )
        print(f"      full_ids length: {full_ids.shape[0]} (prompt={pid.shape[-1]}, ans={len(ans_ids)})")
        with mdl.hooks(hooks or []):
            logits = mdl(full_ids[:-1].unsqueeze(0), return_type="logits")
        logprobs = F.log_softmax(logits.float(), dim=-1)
        tgt = full_ids[1:].unsqueeze(0)
        return logprobs.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    
    baseline_ll = answer_logprob(model, prompt_ids, ans)
    print(f"   baseline_ll = {baseline_ll:.6f}")
    
    hooks = [(f"blocks.{layer}.hook_resid_pre", patch_hook)]
    patched_ll = answer_logprob(model, prompt_ids, ans, hooks=hooks)
    print(f"   patched_ll = {patched_ll:.6f}")
    print(f"   delta = {patched_ll - baseline_ll:.8f}")

# 9. Check hook name validity
print("\n9. Listing TransformerLens hook points with 'resid'...")
hook_names = [name for name, _ in model.hook_dict.items() if "resid" in name]
for name in hook_names[:10]:
    print(f"     {name}")

# 10. FIX TEST: Use clone instead of in-place modification
print("\n10. Testing FIX: clone + return instead of in-place...")
def patch_hook_fixed(resid, hook):
    new_resid = resid.clone()
    new_resid[:, -1, :] = patched_act.to(resid.device)
    return new_resid

with torch.no_grad():
    baseline_logits2 = model(prompt_ids, return_type="logits")
    baseline_last2 = baseline_logits2[0, -1, :].clone()

try:
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook_fixed)]):
        patched_logits3 = model(prompt_ids, return_type="logits")
        patched_last3 = patched_logits3[0, -1, :].clone()
    
    logit_diff3 = (patched_last3 - baseline_last2).abs()
    print(f"   FIXED logit diff at last pos: max={logit_diff3.max().item():.6f}, mean={logit_diff3.mean().item():.6f}")
except Exception as e:
    print(f"   ERROR: {e}")

# 11. Also test torch.cat approach (avoid setitem entirely)
print("\n11. Testing torch.cat approach (no setitem)...")
def patch_hook_cat(resid, hook):
    return torch.cat([
        resid[:, :-1, :],
        patched_act.unsqueeze(0).unsqueeze(0).to(resid.device)
    ], dim=1)

try:
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook_cat)]):
        patched_logits4 = model(prompt_ids, return_type="logits")
        patched_last4 = patched_logits4[0, -1, :].clone()
    
    logit_diff4 = (patched_last4 - baseline_last2).abs()
    print(f"   CAT logit diff at last pos: max={logit_diff4.max().item():.6f}, mean={logit_diff4.mean().item():.6f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
