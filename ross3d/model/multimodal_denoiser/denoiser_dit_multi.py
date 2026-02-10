import contextlib
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import repeat
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from ross3d.model.multimodal_denoiser.diffusion_utils import create_diffusion
from ross3d.utils import rank0_print


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._debug_hooked = False

    def _maybe_register_debug_hooks(self) -> None:
        if self._debug_hooked or os.getenv("ROSS3D_DEBUG_TEMBED") != "1":
            return
        self._debug_hooked = True

        def _log_once(tag: str, message: str) -> None:
            key = f"_ross3d_tembed_{tag}"
            if getattr(self, key, False):
                return
            setattr(self, key, True)
            rank0_print(f"[mm_inv_projector][t_embedder][{tag}] {message}")

        def _make_fwd_hook(tag: str):
            def _hook(module, inputs, output):
                inp = inputs[0] if inputs else None
                _log_once(
                    f"{tag}_fwd",
                    "input="
                    f"{tuple(inp.shape) if inp is not None else None} "
                    f"dtype={getattr(inp, 'dtype', None)} "
                    "output="
                    f"{tuple(output.shape) if output is not None else None} "
                    f"dtype={getattr(output, 'dtype', None)}",
                )
            return _hook

        def _make_bwd_hook(tag: str):
            def _hook(module, grad_input, grad_output):
                gout = grad_output[0] if grad_output else None
                _log_once(
                    f"{tag}_bwd",
                    "grad_output="
                    f"{tuple(gout.shape) if gout is not None else None} "
                    f"dtype={getattr(gout, 'dtype', None)}",
                )
            return _hook

        def _make_weight_grad_hook(tag: str):
            def _hook(grad):
                _log_once(
                    f"{tag}_wgrad",
                    f"weight_grad={tuple(grad.shape)} dtype={grad.dtype}",
                )
                return None
            return _hook

        def _make_bias_grad_hook(tag: str):
            def _hook(grad):
                _log_once(
                    f"{tag}_bgrad",
                    f"bias_grad={tuple(grad.shape)} dtype={grad.dtype}",
                )
                return None
            return _hook

        def _make_grad_shape_hook(tag: str):
            def _hook(grad):
                _log_once(
                    f"{tag}_grad_shape",
                    f"grad_shape={tuple(grad.shape)} dtype={grad.dtype}",
                )
                return None
            return _hook

        def _make_tensor_grad_hook(tag: str):
            def _hook(grad):
                _log_once(
                    f"{tag}_grad",
                    f"grad={tuple(grad.shape)} dtype={grad.dtype}",
                )
                return None
            return _hook

        def _make_param_grad_hook(tag: str, param_shape: tuple):
            def _hook(grad):
                if grad is None:
                    _log_once(f"{tag}_pgrad", "grad=None")
                    return None
                if tuple(grad.shape) != param_shape:
                    _log_once(
                        f"{tag}_pgrad",
                        f"grad_shape={tuple(grad.shape)} expected={param_shape} dtype={grad.dtype}",
                    )
                else:
                    _log_once(
                        f"{tag}_pgrad",
                        f"grad_shape={tuple(grad.shape)} dtype={grad.dtype}",
                    )
                return None
            return _hook

        self.mlp[0].register_forward_hook(_make_fwd_hook("linear1"))
        self.mlp[0].register_full_backward_hook(_make_bwd_hook("linear1"))
        self.mlp[0].weight.register_hook(_make_weight_grad_hook("linear1"))
        self.mlp[0].weight.register_hook(_make_grad_shape_hook("mlp0_weight"))
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.register_hook(_make_bias_grad_hook("linear1"))
            self.mlp[0].bias.register_hook(_make_grad_shape_hook("mlp0_bias"))
        self.mlp[2].register_forward_hook(_make_fwd_hook("linear2"))
        self.mlp[2].register_full_backward_hook(_make_bwd_hook("linear2"))
        self.mlp[2].weight.register_hook(_make_weight_grad_hook("linear2"))
        if self.mlp[2].bias is not None:
            self.mlp[2].bias.register_hook(_make_bias_grad_hook("linear2"))
            self.mlp[2].bias.register_hook(_make_grad_shape_hook("mlp2_bias"))

        for name, param in self.mlp.named_parameters():
            param.register_hook(_make_param_grad_hook(f"mlp_{name}", tuple(param.shape)))

        self._debug_tensor_grad_hook = _make_tensor_grad_hook

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self._maybe_register_debug_hooks()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if os.getenv("ROSS3D_DEBUG_TEMBED") == "1" and t_freq.requires_grad:
            t_freq.register_hook(self._debug_tensor_grad_hook("t_freq"))
        elif os.getenv("ROSS3D_DEBUG_TEMBED") == "1":
            rank0_print(
                "[mm_inv_projector][t_embedder][t_freq] "
                f"requires_grad=False shape={tuple(t_freq.shape)} dtype={t_freq.dtype}"
            )
        if os.getenv("ROSS3D_DEBUG_TEMBED") == "1":
            lin1_out = self.mlp[0](t_freq)
            lin1_out.retain_grad()
            if lin1_out.requires_grad:
                lin1_out.register_hook(self._debug_tensor_grad_hook("linear1_out"))
            else:
                rank0_print(
                    "[mm_inv_projector][t_embedder][linear1_out] "
                    f"requires_grad=False shape={tuple(lin1_out.shape)} dtype={lin1_out.dtype}"
                )
            act_out = self.mlp[1](lin1_out)
            act_out.retain_grad()
            if act_out.requires_grad:
                act_out.register_hook(self._debug_tensor_grad_hook("linear1_act"))
            else:
                rank0_print(
                    "[mm_inv_projector][t_embedder][linear1_act] "
                    f"requires_grad=False shape={tuple(act_out.shape)} dtype={act_out.dtype}"
                )
            lin2_out = self.mlp[2](act_out)
            lin2_out.retain_grad()
            if lin2_out.requires_grad:
                lin2_out.register_hook(self._debug_tensor_grad_hook("linear2_out"))
            else:
                rank0_print(
                    "[mm_inv_projector][t_embedder][linear2_out] "
                    f"requires_grad=False shape={tuple(lin2_out.shape)} dtype={lin2_out.dtype}"
                )
            self._debug_saved_tensors = (lin1_out, act_out, lin2_out)
            t_emb = lin2_out
        else:
            t_emb = self.mlp(t_freq)
        if os.getenv("ROSS3D_DEBUG_TEMBED") == "1" and t_emb.requires_grad:
            def _t_emb_hook(grad):
                lin1_out, act_out, lin2_out = getattr(self, "_debug_saved_tensors", (None, None, None))
                rank0_print(
                    "[mm_inv_projector][t_embedder][saved_grads] "
                    f"lin1_out={getattr(lin1_out, 'grad', None)} "
                    f"linear1_act={getattr(act_out, 'grad', None)} "
                    f"lin2_out={getattr(lin2_out, 'grad', None)}"
                )
                return None
            t_emb.register_hook(_t_emb_hook)
            t_emb.register_hook(self._debug_tensor_grad_hook("t_emb"))
        elif os.getenv("ROSS3D_DEBUG_TEMBED") == "1":
            rank0_print(
                "[mm_inv_projector][t_embedder][t_emb] "
                f"requires_grad=False shape={tuple(t_emb.shape)} dtype={t_emb.dtype}"
            )
        if os.getenv("ROSS3D_DEBUG_TEMBED") == "1":
            rank0_print(
                "[mm_inv_projector][t_embedder] "
                f"t={tuple(t.shape)} dtype={t.dtype} "
                f"t_freq={tuple(t_freq.shape)} dtype={t_freq.dtype} "
                f"t_emb={tuple(t_emb.shape)} dtype={t_emb.dtype}"
            )
        return t_emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        dim: int,
        out_features: int,
        num_heads: int = 16,
        num_queries: int = 196,
        qkv_bias: bool = False,
        linear_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, dim, bias=True),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(0, 1).unsqueeze(0) # all tokens need to be KV
        B, N, C = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls

        out = self.linear(x_cls)
        return out


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size_bev=27,
        input_size_view=14,
        patch_size=1,
        in_channels=64,
        hidden_size=1024,
        z_channel=3584,
        depth=3,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder_bev = PatchEmbed(input_size_bev, patch_size, in_channels, hidden_size, bias=True)
        self.pos_embed_bev = nn.Parameter(torch.zeros(1, self.x_embedder_bev.num_patches, hidden_size), requires_grad=False)
        self.z_embedder_bev = AttentionPoolingClassifier(
            in_features=z_channel,
            dim=hidden_size,
            out_features=hidden_size,
            num_queries=self.x_embedder_bev.num_patches,
            average_pool=False,
        )

        self.x_embedder_view = PatchEmbed(input_size_view, patch_size, in_channels, hidden_size, bias=True)
        # Will use fixed sin-cos embedding:
        self.pos_embed_view = nn.Parameter(torch.zeros(1, self.x_embedder_view.num_patches, hidden_size), requires_grad=False)
        self.z_embedder_view = AttentionPoolingClassifier(
            in_features=z_channel,
            dim=hidden_size,
            out_features=hidden_size,
            num_queries=self.x_embedder_view.num_patches,
            average_pool=False,
        )

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_bev.shape[-1], int(self.x_embedder_bev.num_patches ** 0.5))
        self.pos_embed_bev.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_view.shape[-1], int(self.x_embedder_view.num_patches ** 0.5))
        self.pos_embed_view.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder_bev.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_bev.proj.bias, 0)
        w = self.x_embedder_view.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_view.proj.bias, 0)

        # Initialize condition embedding:
        nn.init.normal_(self.z_embedder_bev.k.weight, std=0.02)
        nn.init.normal_(self.z_embedder_bev.v.weight, std=0.02)
        nn.init.normal_(self.z_embedder_bev.linear.weight, std=0.02)
        nn.init.normal_(self.z_embedder_view.k.weight, std=0.02)
        nn.init.normal_(self.z_embedder_view.v.weight, std=0.02)
        nn.init.normal_(self.z_embedder_view.linear.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, bev):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder_bev.patch_size[0] if bev else self.x_embedder_view.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, context, bev=False, **_unused_kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, C, H, W) tensor of conditions
        """
        z = context.flatten(2).transpose(1, 2).contiguous()
        if bev:
            x = self.x_embedder_bev(x) + self.pos_embed_bev     # (N, T, D), where T = H * W / patch_size ** 2
            z = self.z_embedder_bev(z)  # (N, T, D)
        else:
            x = self.x_embedder_view(x) + self.pos_embed_view  # (N, T, D), where T = H * W / patch_size ** 2
            z = self.z_embedder_view(z)  # (N, T, D)

        t = self.t_embedder(t)      # (N, D)
        c = t.unsqueeze(1) + z      # (N, T, D)
        for block in self.blocks:
            x = block(x, c)         # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, bev) # (N, out_channels, H, W)
        return x


class Ross3DDenoiserMulti(nn.Module):
    def __init__(
        self,
        x_channel,
        z_channel,
        embed_dim,
        depth,
        learn_sigma=False,
        timesteps='1000',
        n_patches_bev=4096,
        n_patches_view=196,
    ):
        super().__init__()
        self.in_channels = x_channel

        self.ln_pre = nn.LayerNorm(z_channel, elementwise_affine=False)
        # self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, z_channel), requires_grad=True)
        # torch.nn.init.normal_(self.pos_embed, std=.02)

        self.net = DiT(
            input_size_bev=int(math.sqrt(n_patches_bev)),
            input_size_view=int(math.sqrt(n_patches_view)),
            patch_size=1,
            in_channels=x_channel,
            hidden_size=embed_dim,
            z_channel=z_channel,
            depth=depth,
            learn_sigma=learn_sigma,
        )
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine", learn_sigma=learn_sigma)
        self.gen_diffusion = create_diffusion(timestep_respacing=timesteps, noise_schedule="cosine", learn_sigma=learn_sigma)
        self._debug_param_hooks_registered = False
        self._debug_param_hook_seen = {}

    def _maybe_register_param_debug_hooks(self) -> None:
        if self._debug_param_hooks_registered or os.getenv("ROSS3D_DEBUG_PARAM_GRADS") != "1":
            return
        self._debug_param_hooks_registered = True
        zero_grad_enabled = os.getenv("ROSS3D_DEBUG_ZERO_GRAD") == "1"
        zero_grad_logged = {"seen": False}

        def _log_once(tag: str, message: str) -> None:
            if self._debug_param_hook_seen.get(tag, False):
                return
            self._debug_param_hook_seen[tag] = True
            rank0_print(f"[mm_inv_projector][param_hook][{tag}] {message}")

        def _describe_tensor(value):
            if value is None:
                return "None"
            if not torch.is_tensor(value):
                return f"{type(value)}"
            return f"shape={tuple(value.shape)} dtype={value.dtype}"

        def _log_zero_grad_buffer(param_name: str, param) -> None:
            if not zero_grad_enabled or zero_grad_logged["seen"]:
                return
            if "t_embedder.mlp.2.weight" in param_name:
                return
            if not hasattr(param, "ds_id"):
                return
            zero_grad_logged["seen"] = True
            rank0_print(
                "[mm_inv_projector][zero_grad_buffer] "
                f"name={param_name} ds_id={getattr(param, 'ds_id', None)} "
                f"ds_status={getattr(param, 'ds_status', None)} "
                f"ds_shape={getattr(param, 'ds_shape', None)} "
                f"ds_numel={getattr(param, 'ds_numel', None)} "
                f"ds_tensor={_describe_tensor(getattr(param, 'ds_tensor', None))} "
                f"ds_grad={_describe_tensor(getattr(param, 'ds_grad', None))} "
                f"ds_primary_gradient={_describe_tensor(getattr(param, 'ds_primary_gradient', None))}"
            )

        def _make_param_grad_hook(param_name: str, param_shape: tuple, param):
            def _hook(grad):
                if grad is None:
                    _log_once(param_name, "grad=None")
                    return None
                grad_shape = tuple(grad.shape)
                if grad_shape != param_shape:
                    _log_once(
                        param_name,
                        f"grad_shape={grad_shape} expected={param_shape} dtype={grad.dtype}",
                    )
                else:
                    _log_once(
                        param_name,
                        f"grad_shape={grad_shape} dtype={grad.dtype}",
                    )
                _log_zero_grad_buffer(param_name, param)
                return None
            return _hook

        matching_1024 = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            param.register_hook(_make_param_grad_hook(name, tuple(param.shape), param))
            if tuple(param.shape) == (1024, 1024):
                matching_1024.append(name)

        if matching_1024:
            rank0_print(
                "[mm_inv_projector][param_hook][shape_1024x1024] "
                f"params={', '.join(matching_1024)}"
            )

    @contextlib.contextmanager
    def _maybe_disable_checkpoint(self):
        if os.getenv("ROSS3D_NO_DENOISER_CKPT") != "1":
            yield
            return
        import torch.utils.checkpoint as checkpoint_module

        original_checkpoint = checkpoint_module.checkpoint
        patched_modules = []
        log_once = {"seen": False}

        def _no_checkpoint(function, *args, **kwargs):
            if not log_once["seen"]:
                log_once["seen"] = True
                rank0_print("[mm_inv_projector][ckpt] checkpoint bypassed")
            return function(*args, **kwargs)

        checkpoint_module.checkpoint = _no_checkpoint
        for module in sys.modules.values():
            if module is None:
                continue
            if getattr(module, "checkpoint", None) is original_checkpoint:
                patched_modules.append((module, original_checkpoint))
                setattr(module, "checkpoint", _no_checkpoint)
        try:
            yield
        finally:
            checkpoint_module.checkpoint = original_checkpoint
            for module, checkpoint_ref in patched_modules:
                setattr(module, "checkpoint", checkpoint_ref)

    def forward(self, z, target, bev):
        # z: [B, C, H, W] output features
        # x: [B, C, H, W] clean latent features
        self._maybe_register_param_debug_hooks()
        t = torch.randint(self.train_diffusion.num_timesteps, size=(target.shape[0],), device=target.device).long()
        debug_shapes = os.getenv("ROSS3D_DEBUG_DENOISER_SHAPES") == "1"
        model_kwargs = dict(context=z, bev=bev, debug_denoiser_shapes=debug_shapes)
        no_grad_denoiser = os.getenv("ROSS3D_DENOISER_NO_GRAD") == "1"
        with self._maybe_disable_checkpoint():
            if no_grad_denoiser:
                with torch.no_grad():
                    loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
            else:
                loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict.get("loss_scalar", loss_dict["loss"])
        if loss.numel() != 1:
            rank0_print(
                "[mm_inv_projector][denoiser] "
                f"loss is not scalar; shape={tuple(loss.shape)}. "
                "Reducing with mean()."
            )
            loss = loss.mean()
        if no_grad_denoiser:
            loss = loss.detach()

        return loss

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0, bev=False, hw=27):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, hw, hw).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(context=z, cfg_scale=cfg, bev=bev)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, hw, hw).cuda()
            model_kwargs = dict(context=z, bev=bev)
            sample_fn = self.net.forward

        sampled_token_latent = self.train_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=True, model_kwargs=model_kwargs, progress=True,
            temperature=temperature
        )

        return sampled_token_latent


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    model = RossDenoiserMulti(
        x_channel=64,
        z_channel=3584,
        embed_dim=1024,
        depth=3,
        n_patches_bev=4096,
        n_patches_view=196,
    )
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6
    print(total_params, "M")

    x = torch.randn([1, 64, 64, 64]).cuda()
    z = torch.randn([32, 3584, 14, 14]).cuda()

    with torch.inference_mode():
        loss = model(z, x)

    print(loss.shape, loss.mean())
