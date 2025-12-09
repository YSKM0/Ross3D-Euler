import math
import torch

from ross3d.model import Ross3DQwenForCausalLM  # adjust path if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the full Ross3D Qwen model (same checkpoint as in your train script)
model_path = "/cluster/scratch/hanwliu/projects/ross3d/models/llava-video-qwen2-7b-ross3d"

model = Ross3DQwenForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,   # use fp32 for debugging to avoid weird NaNs
    low_cpu_mem_usage=False,
)
model.to(DEVICE)
model.eval()

# 'model' is the meta wrapper (Ross3DQwenForCausalLM / Ross3DMetaForCausalLM)
# The inner LM body is model.model (Ross3DQwenModel)

inner = model.model  # just for convenience if you need it

# 2. TEMP: spoof tiny image_embed_len so we can hand-craft indices
inner.image_embed_len = 4   # P = 4 patches per frame
patch_h = int(math.ceil(math.sqrt(inner.image_embed_len)))  # = 2

hidden_dim = model.config.hidden_size

# 3. Build synthetic hidden_states + indices consistent with your slicing logic
#
# Recall your code per frame:
#   cur_hidden_states = [hs[boi : newline[f*patch_h]]]
#   for k in range(f*patch_h+1, (f+1)*patch_h):
#       cur_hidden_states.append(hs[newline[k-1]+1 : newline[k]])
#   cur_hidden_states.append(hs[newline[(f+1)*patch_h-1]+1 : eoi])
#
# We want len(cur_hidden_states[0]) + len(cur_hidden_states[1]) + len(cur_hidden_states[2]) = image_embed_len (=4)
# Choose lengths: 2,1,1 → 4.

num_frames = 3
tokens_per_frame_span = 7  # [boi, 2 tok, newline0, 1 tok, newline1, 1 tok, eoi]
total_seq_len = num_frames * tokens_per_frame_span  # 21

boi_ids = []
eoi_ids = []
newline_ids = []

for f in range(num_frames):
    base = f * tokens_per_frame_span

    boi = base + 0
    newline0 = base + 2  # boi + 2
    newline1 = base + 4  # newline0 + 1 + 1
    eoi = base + 6       # newline1 + 1 + 1

    boi_ids.append(boi)
    eoi_ids.append(eoi)

    # order must match your usage: newline_ids[f*patch_h], newline_ids[f*patch_h+1]
    newline_ids.append(newline0)
    newline_ids.append(newline1)

boi_ids_list = boi_ids               # function expects List[int]
eoi_ids_list = eoi_ids
newline_ids_tensor = torch.LongTensor(newline_ids).to(DEVICE)

# Random hidden states [B=1, L=total_seq_len, D]
hidden_states = torch.randn(
    1, total_seq_len, hidden_dim,
    device=DEVICE,
    requires_grad=True,
)

mask = None  # no masked frames for this sanity check

# 4. Call your cycle-consistency loss on the *outer* model
with torch.set_grad_enabled(True):
    cycle_loss = model.compute_cycle_consistency_loss(
        hidden_states=hidden_states,
        boi_ids=boi_ids_list,
        eoi_ids=eoi_ids_list,
        newline_ids=linenewline_ids_tensor,
        mask=mask,
        num_walks=None,          # use all visible frames
        temperature=0.07,
    )

print("cycle_loss:", cycle_loss.item())
print("requires_grad:", cycle_loss.requires_grad)

# 5. Backprop to verify gradients flow
cycle_loss.backward()
print("grad on hidden_states:", hidden_states.grad is not None)
print("grad shape:", None if hidden_states.grad is None else hidden_states.grad.shape)
