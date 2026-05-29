import torch
import math
from MeMoPyTorch.modelling_memo import MeMo
from MeMoPyTorch.modelling_memo_tokenizer import MeMoTokenizer

# ── Parameters ────────────────────────────────────────────────────────────
# We use small d but bigger than before for better vector separation
d, h, l = 128, 2, 2
chunk_length = 4   # h^l = 4

torch.manual_seed(42)

# ── Build tokenizer and model ─────────────────────────────────────────────
tokenizer = MeMoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b",
    padding_side='left', truncation_side='left',
    max_length=chunk_length, head_number=h
)
tokenizer.pad_token = tokenizer.eos_token

model = MeMo(
    inner_dim=d, num_of_heads=h, num_of_layers=l,
    chunk_length=chunk_length,
    num_embeddings=tokenizer.vocab_size,
    padding_idx=tokenizer.pad_token_id,
    device='cpu'
)

last_layer = model.layers[l-1]

print("Model built ✓")
print(f"Last layer CMM shape: {last_layer.CMM.weight.shape}")
print(f"CMM initially zeros: {torch.all(last_layer.CMM.weight == 0).item()}")

# ── The 7 inference patterns (all correct answer = NO) ────────────────────
patterns = {
    "ASd": "From 'If the match is struck, then it will light', can we infer 'If the match is struck and it has been soaked in water, then it will light'?",
    "CMP": "Suppose that the Lakers, Warriors, and Celtics have the best odds to win the NBA championship. Based on expert projections, it is most likely the Lakers will win, followed closely by the Warriors, with the Celtics having much lower odds. These are the only top contenders, so if the Warriors don't win, then if the Lakers don't win, the Celtics will win. Moreover, it is most likely the Lakers will win and hence that the Warriors won't win. Does it follow that it is likely that, if the Lakers don't win, the Celtics will?",
    "CT":  "From 'If Mary was at the wedding, then Sue was at the wedding', can we infer 'If Sue was not at the wedding, then Mary was not at the wedding'?",
    "DSmi":"From 'Either Fido is inside or Fido must be in the garden' together with 'Fido might not be in the garden', can we infer 'Fido is inside'?",
    "DSmu":"From 'Either Fido is inside or Fido must be in the garden' together with 'It's not the case that Fido must be in the garden', can we infer 'Fido is inside'?",
    "MTmi":"From 'If Mary was at the wedding, then Sue must have been at the wedding' together with 'Sue might not have been at the wedding', can we infer 'Mary was not at the wedding'?",
    "MTmu":"From 'If Mary was at the wedding, then Sue must have been at the wedding' together with 'It's not the case that Sue must have been at the wedding', can we infer 'Mary was not at the wedding'?",
}

# ── A clearly VALID inference (correct answer = YES) ─────────────────────
# Standard Modus Tollens — no modals, should be valid
valid_pattern = "From 'If Mary was at the wedding, then Sue was at the wedding' together with 'Sue was not at the wedding', can we infer 'Mary was not at the wedding'?"

# ── Step 1: Encode each pattern into a d-dimensional key vector ───────────
# We use MeMo's own encoder (the embedding layer) to get token embeddings
# Then average them to get one vector per pattern
# This is the simplest way to get a real linguistic representation

print("\n── Step 1: Encoding patterns into vectors ──")

def text_to_vector(text, model, tokenizer, d):
    """
    Convert text to a single d-dimensional vector using MeMo's own embeddings.
    We tokenize, embed, and average the token vectors.
    """
    # Tokenize — take first chunk_length tokens
    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=chunk_length+1
    )
    input_ids = tokens['input_ids'][..., :-1]  # shape: (1, chunk_length)
    
    # Embed using MeMo's encoder
    with torch.no_grad():
        embedded = model.encoder.encode(input_ids)  # shape: (1, chunk_length, d)
        # Average over token dimension to get one vector
        vec = embedded.mean(dim=1).squeeze(0)        # shape: (d,)
        # Normalize
        vec = vec / (vec.norm() + 1e-8)
    return vec

# Encode all 7 invalid patterns
pattern_vectors = {}
for name, text in patterns.items():
    vec = text_to_vector(text, model, tokenizer, d)
    pattern_vectors[name] = vec
    print(f"  {name}: vector norm = {vec.norm():.4f}")

# Encode the valid pattern
valid_vec = text_to_vector(valid_pattern, model, tokenizer, d)
print(f"  MT_valid: vector norm = {valid_vec.norm():.4f}")

# ── Step 2: Create answer vectors ─────────────────────────────────────────
# VALUE_NO  = what we want MeMo to retrieve for invalid patterns
# VALUE_YES = what we want MeMo to retrieve for valid patterns
# These are fixed random vectors representing "NO" and "YES"

torch.manual_seed(100)
value_NO  = torch.randn(d); value_NO  = value_NO  / value_NO.norm()
value_YES = torch.randn(d); value_YES = value_YES / value_YES.norm()

print(f"\n  NO/YES vector similarity: {torch.dot(value_NO, value_YES):.4f}")
print(f"  (Should be ~0 — they represent opposite answers)")

# ── Step 3: Write rules into MeMo's CMM ──────────────────────────────────
print("\n── Step 2: Writing rules into CMM ──")

# Write all 7 invalid patterns → VALUE_NO
for name, vec in pattern_vectors.items():
    rule = torch.outer(vec, value_NO)  # outer product: key ⊗ value
    last_layer.CMM.memorize(rule)
    print(f"  Written: {name} → NO")

# Write the valid pattern → VALUE_YES
rule_valid = torch.outer(valid_vec, value_YES)
last_layer.CMM.memorize(rule_valid)
print(f"  Written: MT_valid → YES")

# ── Step 4: Query and check retrieval ─────────────────────────────────────
print("\n── Step 3: Querying CMM with each pattern ──")
print(f"{'Pattern':<10} {'→NO sim':>10} {'→YES sim':>10} {'Correct?':>10}")
print("-" * 45)

all_correct = True

for name, vec in pattern_vectors.items():
    with torch.no_grad():
        retrieved = last_layer.CMM(vec.unsqueeze(0)).squeeze(0)
    
    sim_no  = torch.dot(retrieved, value_NO).item()
    sim_yes = torch.dot(retrieved, value_YES).item()
    correct = sim_no > sim_yes
    if not correct:
        all_correct = False
    
    print(f"{name:<10} {sim_no:>10.4f} {sim_yes:>10.4f} {'✓' if correct else '✗':>10}")

# Check valid pattern
with torch.no_grad():
    retrieved_valid = last_layer.CMM(valid_vec.unsqueeze(0)).squeeze(0)
sim_no  = torch.dot(retrieved_valid, value_NO).item()
sim_yes = torch.dot(retrieved_valid, value_YES).item()
correct_valid = sim_yes > sim_no
print(f"{'MT_valid':<10} {sim_no:>10.4f} {sim_yes:>10.4f} {'✓' if correct_valid else '✗':>10}")

print("\n" + "="*45)
if all_correct and correct_valid:
    print("ALL PATTERNS CORRECTLY STORED AND RETRIEVED ✓")
    print("→ MeMo can distinguish valid from invalid inference by design")
else:
    print("Some patterns not correctly retrieved — see above")

# ── Step 5: Similarity between pattern vectors ────────────────────────────
# This tells us: how well does MeMo's embedding separate the 7 patterns?
# If two patterns have high similarity, their keys overlap and may interfere

print("\n── Step 4: How similar are the pattern vectors to each other? ──")
print("(High similarity = patterns may interfere in CMM)")
print()

names = list(pattern_vectors.keys()) + ["MT_valid"]
all_vecs = [pattern_vectors[n] for n in names[:-1]] + [valid_vec]

for i in range(len(names)):
    for j in range(i+1, len(names)):
        sim = torch.dot(all_vecs[i], all_vecs[j]).item()
        flag = " ← HIGH" if abs(sim) > 0.5 else ""
        print(f"  {names[i]:<10} vs {names[j]:<10}: {sim:>7.4f}{flag}")

# ── Step 6: Test with UNSEEN text ─────────────────────────────────────────
# Does MeMo stay silent for text it has never seen?
print("\n── Step 5: Testing with unseen/unrelated text ──")

unseen_texts = [
    "The cat sat on the mat.",                          # totally unrelated
    "If it rains then the ground gets wet.",            # simple valid conditional
    "What is the capital of France?",                   # factual question
]

for text in unseen_texts:
    vec = text_to_vector(text, model, tokenizer, d)
    with torch.no_grad():
        retrieved = last_layer.CMM(vec.unsqueeze(0)).squeeze(0)
    sim_no  = torch.dot(retrieved, value_NO).item()
    sim_yes = torch.dot(retrieved, value_YES).item()
    print(f"  '{text[:50]}...' " if len(text) > 50 else f"  '{text}'")
    print(f"    → NO: {sim_no:.4f}  YES: {sim_yes:.4f}  (neither should dominate)")