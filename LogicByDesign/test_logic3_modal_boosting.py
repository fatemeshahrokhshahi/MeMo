import torch
from MeMoPyTorch.modelling_memo import MeMo
from MeMoPyTorch.modelling_memo_tokenizer import MeMoTokenizer

# ── Same setup ────────────────────────────────────────────────────────────
d, h, l = 128, 2, 2
chunk_length = 4
torch.manual_seed(42)

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

# ── Modal-aware encoding ──────────────────────────────────────────────────
# Key insight from your LogiCue and predictability papers:
# "must", "might", "not" are the semantically critical words
# We give them higher weight in the key vector

MODAL_KEYWORDS = ["must", "might", "not", "necessarily", "possibly", "case"]

def get_token_weight(token_id, tokenizer, boost=5.0):
    """Give higher weight to modal/negation tokens."""
    token_str = tokenizer.decode([token_id]).strip().lower()
    for keyword in MODAL_KEYWORDS:
        if keyword in token_str:
            return boost
    return 1.0

def text_to_modal_vector(text, model, tokenizer, d, boost=5.0):
    """
    Convert text to vector with modal keywords boosted.
    Uses the FULL text (not truncated) to capture modal words anywhere.
    """
    # Tokenize full text without truncation
    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=False,
        truncation=False,
    )
    input_ids = tokens['input_ids'][0]  # shape: (seq_len,)

    # Embed all tokens
    with torch.no_grad():
        # Get embeddings directly from the embedding layer
        embedded = model.encoder(input_ids)  # shape: (seq_len, d)

    # Compute weights for each token
    weights = torch.tensor([
        get_token_weight(tid.item(), tokenizer, boost)
        for tid in input_ids
    ], dtype=torch.float)

    # Weighted average
    weights = weights / weights.sum()
    vec = (embedded * weights.unsqueeze(1)).sum(dim=0)  # shape: (d,)
    vec = vec / (vec.norm() + 1e-8)
    return vec

# ── Patterns ──────────────────────────────────────────────────────────────
patterns = {
    "ASd":  "From 'If the match is struck, then it will light', can we infer 'If the match is struck and it has been soaked in water, then it will light'?",
    "CMP":  "Suppose that the Lakers, Warriors, and Celtics have the best odds to win the NBA championship. Based on expert projections, it is most likely the Lakers will win, followed closely by the Warriors, with the Celtics having much lower odds. These are the only top contenders, so if the Warriors don't win, then if the Lakers don't win, the Celtics will win. Moreover, it is most likely the Lakers will win and hence that the Warriors won't win. Does it follow that it is likely that, if the Lakers don't win, the Celtics will?",
    "CT":   "From 'If Mary was at the wedding, then Sue was at the wedding', can we infer 'If Sue was not at the wedding, then Mary was not at the wedding'?",
    "DSmi": "From 'Either Fido is inside or Fido must be in the garden' together with 'Fido might not be in the garden', can we infer 'Fido is inside'?",
    "DSmu": "From 'Either Fido is inside or Fido must be in the garden' together with 'It is not the case that Fido must be in the garden', can we infer 'Fido is inside'?",
    "MTmi": "From 'If Mary was at the wedding, then Sue must have been at the wedding' together with 'Sue might not have been at the wedding', can we infer 'Mary was not at the wedding'?",
    "MTmu": "From 'If Mary was at the wedding, then Sue must have been at the wedding' together with 'It is not the case that Sue must have been at the wedding', can we infer 'Mary was not at the wedding'?",
}
valid_pattern = "From 'If Mary was at the wedding, then Sue was at the wedding' together with 'Sue was not at the wedding', can we infer 'Mary was not at the wedding'?"

# ── Encode with modal boosting ────────────────────────────────────────────
print("── Modal-aware encoding ──")
pattern_vectors = {}
for name, text in patterns.items():
    vec = text_to_modal_vector(text, model, tokenizer, d)
    pattern_vectors[name] = vec
    print(f"  {name}: norm={vec.norm():.4f}")

valid_vec = text_to_modal_vector(valid_pattern, model, tokenizer, d)
print(f"  MT_valid: norm={valid_vec.norm():.4f}")

# ── Check similarity now ──────────────────────────────────────────────────
print("\n── Key similarities after modal boosting ──")
names = list(pattern_vectors.keys()) + ["MT_valid"]
all_vecs = list(pattern_vectors.values()) + [valid_vec]

problem_found = False
for i in range(len(names)):
    for j in range(i+1, len(names)):
        sim = torch.dot(all_vecs[i], all_vecs[j]).item()
        flag = " ← STILL HIGH" if abs(sim) > 0.5 else " ✓"
        if abs(sim) > 0.5:
            problem_found = True
        print(f"  {names[i]:<10} vs {names[j]:<10}: {sim:>7.4f}{flag}")

if not problem_found:
    print("\n  All pattern pairs are now sufficiently different ✓")
else:
    print("\n  Some patterns still overlap — see above")

# ── Store and retrieve ────────────────────────────────────────────────────
torch.manual_seed(100)
value_NO  = torch.randn(d); value_NO  = value_NO  / value_NO.norm()
value_YES = torch.randn(d); value_YES = value_YES / value_YES.norm()

for name, vec in pattern_vectors.items():
    last_layer.CMM.memorize(torch.outer(vec, value_NO))
last_layer.CMM.memorize(torch.outer(valid_vec, value_YES))

print("\n── Retrieval results with modal-aware keys ──")
print(f"{'Pattern':<12} {'→NO':>8} {'→YES':>8} {'Correct?':>10}")
print("-" * 45)

all_correct = True
for name, vec in pattern_vectors.items():
    with torch.no_grad():
        r = last_layer.CMM(vec.unsqueeze(0)).squeeze(0)
    sno  = torch.dot(r, value_NO).item()
    syes = torch.dot(r, value_YES).item()
    ok   = sno > syes
    if not ok: all_correct = False
    print(f"{name:<12} {sno:>8.4f} {syes:>8.4f} {'✓' if ok else '✗':>10}")

with torch.no_grad():
    r = last_layer.CMM(valid_vec.unsqueeze(0)).squeeze(0)
sno  = torch.dot(r, value_NO).item()
syes = torch.dot(r, value_YES).item()
ok   = syes > sno
if not ok: all_correct = False
print(f"{'MT_valid':<12} {sno:>8.4f} {syes:>8.4f} {'✓' if ok else '✗':>10}")

print("\n" + "="*45)
print("ALL CORRECT ✓" if all_correct else "Some failed — check similarities above")

# ── Show what tokens got boosted ──────────────────────────────────────────
print("\n── Which tokens were boosted for MTmu vs MT_valid? ──")
print("(This shows WHY they are now different)")

for label, text in [("MTmu", patterns["MTmu"]), ("MT_valid", valid_pattern)]:
    tokens_ids = tokenizer(text, return_tensors='pt',
                           padding=False, truncation=False)['input_ids'][0]
    print(f"\n  {label}:")
    for tid in tokens_ids:
        token_str = tokenizer.decode([tid.item()])
        w = get_token_weight(tid.item(), tokenizer, boost=5.0)
        if w > 1.0:
            print(f"    BOOSTED → '{token_str}' (weight={w})")