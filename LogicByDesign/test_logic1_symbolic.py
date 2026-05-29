import torch
import math
from MeMoPyTorch.modelling_memo_layer import MeMoLayer, CompositionOp
from MeMoPyTorch.modelling_memo_embedding import MeMoEmbedding

# ── Tiny parameters ──────────────────────────────────────────────────────
d, h = 64, 2   # dimension, heads
# We use just ONE layer with is_last=True so it has a CMM we can write into

torch.manual_seed(42)

# ── Build one MeMo layer ─────────────────────────────────────────────────
layer = MeMoLayer(
    inner_dim=d,
    num_of_heads=h,
    init_weights=True,
    is_last=True,          # gives it a CMM to store into
    compositionOp=CompositionOp.Prod
)

print("Layer built:", layer)
print("CMM weight shape:", layer.CMM.weight.shape)  # should be (d, d) = (64, 64)
print("CMM initially all zeros:", torch.all(layer.CMM.weight == 0).item())

# ── Create symbolic vectors for our rule ────────────────────────────────
# We'll represent:
#   KEY   = encoding of the sequence [P→□Q, ¬□Q]  (the invalid MTmu premises)
#   VALUE = encoding of "NO / invalid"
#
# In MeMo, both are random d-dimensional vectors (nearly orthogonal)
# This simulates what would happen if we encoded a logical rule directly

torch.manual_seed(1)
key_MTmu_invalid = torch.randn(d)          # represents "MTmu premise pattern"
key_MTmu_invalid = key_MTmu_invalid / key_MTmu_invalid.norm()  # normalize

torch.manual_seed(2)
value_NO = torch.randn(d)                  # represents "NO / invalid conclusion"
value_NO = value_NO / value_NO.norm()

torch.manual_seed(3)
value_YES = torch.randn(d)                 # represents "YES / valid conclusion"
value_YES = value_YES / value_YES.norm()

print(f"\nKey-Value similarity before storing: {torch.dot(key_MTmu_invalid, value_NO):.4f}")
print(f"(Should be ~0 since random vectors are nearly orthogonal)")

# ── Write the rule into the CMM directly ────────────────────────────────
# CMM stores: C += key^T * value  (outer product)
# This is exactly what layer.CMM.memorize() does

rule_update = torch.outer(key_MTmu_invalid, value_NO)  # shape: (d, d)
print(f"\nRule update matrix shape: {rule_update.shape}")

# Write it in (using MeMo's own memorize method)
layer.CMM.memorize(rule_update)
print("Rule written into CMM ✓")
print(f"CMM no longer all zeros: {not torch.all(layer.CMM.weight == 0).item()}")

# ── Query the CMM with the key ───────────────────────────────────────────
# CMM retrieval: output = key @ CMM = key @ (key^T * value) ≈ value
# because key · key ≈ 1 (normalized) and key · other_keys ≈ 0

with torch.no_grad():
    retrieved = layer.CMM(key_MTmu_invalid.unsqueeze(0))  # shape: (1, d)
    retrieved = retrieved.squeeze(0)

print(f"\n── Retrieval test ──")
print(f"  Similarity of retrieved vector to VALUE_NO  : {torch.dot(retrieved, value_NO):.4f}")
print(f"  Similarity of retrieved vector to VALUE_YES : {torch.dot(retrieved, value_YES):.4f}")
print(f"  (If >> 0 for NO and ~0 for YES: rule is stored and retrieved correctly)")

# ── Now test with a DIFFERENT (wrong) query ──────────────────────────────
torch.manual_seed(99)
random_key = torch.randn(d)
random_key = random_key / random_key.norm()

with torch.no_grad():
    retrieved_random = layer.CMM(random_key.unsqueeze(0)).squeeze(0)

print(f"\n── Random key retrieval (should NOT retrieve VALUE_NO) ──")
print(f"  Similarity to VALUE_NO  : {torch.dot(retrieved_random, value_NO):.4f}")
print(f"  (Should be ~0 — CMM should not fire for unrelated queries)")

# ── Store a SECOND rule (MTmi: also invalid) ─────────────────────────────
torch.manual_seed(4)
key_MTmi_invalid = torch.randn(d)
key_MTmi_invalid = key_MTmi_invalid / key_MTmi_invalid.norm()

rule_update_2 = torch.outer(key_MTmi_invalid, value_NO)
layer.CMM.memorize(rule_update_2)
print(f"\n── After storing second rule (MTmi) ──")

with torch.no_grad():
    # Query with MTmu key — should still get NO
    r1 = layer.CMM(key_MTmu_invalid.unsqueeze(0)).squeeze(0)
    # Query with MTmi key — should also get NO  
    r2 = layer.CMM(key_MTmi_invalid.unsqueeze(0)).squeeze(0)

print(f"  MTmu query → similarity to NO : {torch.dot(r1, value_NO):.4f}")
print(f"  MTmi query → similarity to NO : {torch.dot(r2, value_NO):.4f}")
print(f"  (Both should be clearly > 0)")

# ── Interference test ────────────────────────────────────────────────────
# Now store a VALID rule (e.g. MT: from P→Q and ¬Q, conclude ¬P = YES/valid)
torch.manual_seed(5)
key_MT_valid = torch.randn(d)
key_MT_valid = key_MT_valid / key_MT_valid.norm()

rule_update_3 = torch.outer(key_MT_valid, value_YES)
layer.CMM.memorize(rule_update_3)

print(f"\n── After storing a VALID rule (MT→YES) ──")

with torch.no_grad():
    r_mtmu = layer.CMM(key_MTmu_invalid.unsqueeze(0)).squeeze(0)
    r_mt   = layer.CMM(key_MT_valid.unsqueeze(0)).squeeze(0)

print(f"  MTmu query → similarity to NO  : {torch.dot(r_mtmu, value_NO):.4f}")
print(f"  MTmu query → similarity to YES : {torch.dot(r_mtmu, value_YES):.4f}")
print(f"  MT query   → similarity to YES : {torch.dot(r_mt,   value_YES):.4f}")
print(f"  MT query   → similarity to NO  : {torch.dot(r_mt,   value_NO):.4f}")
print(f"\n  (MTmu should still point to NO, MT should point to YES)")
print(f"  (If this works: multiple rules coexist without interference)")