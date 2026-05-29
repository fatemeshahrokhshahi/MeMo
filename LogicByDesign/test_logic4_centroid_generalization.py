import torch
import json
import os
from MeMoPyTorch.modelling_memo import MeMo
from MeMoPyTorch.modelling_memo_tokenizer import MeMoTokenizer

# ── Setup ─────────────────────────────────────────────────────────────────
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
    num_embeddings=len(tokenizer),
    padding_idx=tokenizer.pad_token_id,
    device='cpu'
)
last_layer = model.layers[l-1]

# ── Modal-aware encoding ──────────────────────────────────────────────────
MODAL_KEYWORDS = ["must", "might", "not", "necessarily", "possibly", "case"]

def get_token_weight(token_id, tokenizer, boost=5.0):
    token_str = tokenizer.decode([token_id]).strip().lower()
    for keyword in MODAL_KEYWORDS:
        if keyword in token_str:
            return boost
    return 1.0

def text_to_modal_vector(text, model, tokenizer, boost=5.0):
    tokens = tokenizer(
        text, return_tensors='pt',
        padding=False, truncation=True,
        max_length=256,
)
    input_ids = tokens['input_ids'][0]
    with torch.no_grad():
        embedded = model.encoder(input_ids)   # shape: (seq_len, d)
    weights = torch.tensor([
        get_token_weight(tid.item(), tokenizer, boost)
        for tid in input_ids
    ], dtype=torch.float)
    weights = weights / weights.sum()
    vec = (embedded * weights.unsqueeze(1)).sum(dim=0)
    vec = vec / (vec.norm() + 1e-8)
    return vec

# ── Load all 80 examples per pattern ─────────────────────────────────────
import urllib.request
import json
import os

DATA_BASE_URL = "https://raw.githubusercontent.com/fatemeshahrokhshahi/LogiCue/main/prompt"
pattern_names = ["ASd", "CMP", "CT", "DSmi", "DSmu", "MTmi", "MTmu"]

all_data = {}
for name in pattern_names:
    url = f"{DATA_BASE_URL}/{name}.json"
    with urllib.request.urlopen(url) as response:
        examples = json.load(response)
    all_data[name] = [ex[2] for ex in examples]
    print(f"Loaded {len(all_data[name])} examples for {name}")
    
# ── Answer vectors ────────────────────────────────────────────────────────
torch.manual_seed(100)
value_NO  = torch.randn(d); value_NO  = value_NO  / value_NO.norm()
value_YES = torch.randn(d); value_YES = value_YES / value_YES.norm()

# ── EXPERIMENT: Train on first 40, test on last 40 ────────────────────────
# For each pattern:
#   - Encode examples 1-40 → write key→NO into CMM (training set)
#   - Query with examples 41-80 → check if CMM retrieves NO (test set)
# This tests whether MeMo GENERALIZES within a pattern

SPLIT = 40

print(f"\n{'='*60}")
print(f"EXPERIMENT: Store first {SPLIT} examples, test on last {SPLIT}")
print(f"{'='*60}")

# ── Step 1: Build and store average key per pattern from training set ─────
# We compute the CENTROID (average vector) of the first 40 examples
# This represents the "typical" encoding of each pattern

print("\n── Building pattern centroids from first 40 examples ──")

pattern_centroids = {}
for name in pattern_names:
    train_examples = all_data[name][:SPLIT]
    vecs = []
    for text in train_examples:
        vec = text_to_modal_vector(text, model, tokenizer)
        vecs.append(vec)
    
    # Average all training vectors → centroid
    centroid = torch.stack(vecs).mean(dim=0)
    centroid = centroid / (centroid.norm() + 1e-8)
    pattern_centroids[name] = centroid
    
    # Measure spread: how similar are the training examples to each other?
    sims = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sims.append(torch.dot(vecs[i], vecs[j]).item())
    avg_sim = sum(sims) / len(sims)
    print(f"  {name}: centroid built, avg within-pattern similarity = {avg_sim:.4f}")

# ── Step 2: Write centroids into CMM ─────────────────────────────────────
print("\n── Writing pattern centroids into CMM ──")
for name, centroid in pattern_centroids.items():
    last_layer.CMM.memorize(torch.outer(centroid, value_NO))
    print(f"  Written: {name} → NO")

# ── Step 3: Test on the remaining 40 examples ────────────────────────────
print(f"\n── Testing on last {SPLIT} examples (unseen during storage) ──")
print(f"\n{'Pattern':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Avg NO sim':>12} {'Avg YES sim':>12}")
print("-" * 65)

overall_correct = 0
overall_total = 0

for name in pattern_names:
    test_examples = all_data[name][SPLIT:]
    correct = 0
    no_sims = []
    yes_sims = []
    
    for text in test_examples:
        vec = text_to_modal_vector(text, model, tokenizer)
        with torch.no_grad():
            retrieved = last_layer.CMM(vec.unsqueeze(0)).squeeze(0)
        
        sim_no  = torch.dot(retrieved, value_NO).item()
        sim_yes = torch.dot(retrieved, value_YES).item()
        no_sims.append(sim_no)
        yes_sims.append(sim_yes)
        
        if sim_no > sim_yes:
            correct += 1
    
    total = len(test_examples)
    accuracy = correct / total * 100
    overall_correct += correct
    overall_total += total
    
    print(f"{name:<10} {correct:>8} {total:>8} {accuracy:>9.1f}% "
          f"{sum(no_sims)/len(no_sims):>12.4f} "
          f"{sum(yes_sims)/len(yes_sims):>12.4f}")

overall_acc = overall_correct / overall_total * 100
print("-" * 65)
print(f"{'TOTAL':<10} {overall_correct:>8} {overall_total:>8} {overall_acc:>9.1f}%")

# ── Step 4: Centroid similarity matrix (between patterns) ─────────────────
print(f"\n── Centroid similarity matrix (should be low between patterns) ──")
print(f"{'':10}", end="")
for n in pattern_names:
    print(f"{n:>8}", end="")
print()

for n1 in pattern_names:
    print(f"{n1:<10}", end="")
    for n2 in pattern_names:
        sim = torch.dot(pattern_centroids[n1], pattern_centroids[n2]).item()
        print(f"{sim:>8.3f}", end="")
    print()

# ── Step 5: Within-pattern vs between-pattern similarity ─────────────────
print(f"\n── Key question: Are within-pattern similarities > between-pattern? ──")
print(f"(If yes: MeMo can distinguish the 7 patterns from each other)")

within_sims = []
between_sims = []

for i, n1 in enumerate(pattern_names):
    for j, n2 in enumerate(pattern_names):
        sim = torch.dot(pattern_centroids[n1], pattern_centroids[n2]).item()
        if i == j:
            within_sims.append(sim)
        else:
            between_sims.append(sim)

print(f"  Average within-pattern similarity  (diagonal): {sum(within_sims)/len(within_sims):.4f}")
print(f"  Average between-pattern similarity (off-diag): {sum(between_sims)/len(between_sims):.4f}")
print(f"  Separation ratio: {(sum(within_sims)/len(within_sims)) / (sum(between_sims)/len(between_sims)):.2f}x")
print(f"\n  (Ratio >> 1 means patterns are well separated in MeMo's space)")

# ── Step 6: Does MeMo stay silent for truly unrelated text? ──────────────
print(f"\n── Specificity test: unrelated text should NOT retrieve NO ──")

unrelated = [
    "The cat sat on the mat.",
    "What is the capital of France?",
    "Two plus two equals four.",
    "She opened the window and looked outside.",
]

for text in unrelated:
    vec = text_to_modal_vector(text, model, tokenizer)
    with torch.no_grad():
        retrieved = last_layer.CMM(vec.unsqueeze(0)).squeeze(0)
    sim_no  = torch.dot(retrieved, value_NO).item()
    sim_yes = torch.dot(retrieved, value_YES).item()
    verdict = "FIRES (unexpected)" if sim_no > 1.0 else "silent ✓"
    print(f"  '{text[:55]}' → NO:{sim_no:.3f} {verdict}")