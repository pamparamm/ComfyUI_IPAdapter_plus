import copy
import math
from typing import Mapping, Optional

import comfy.model_management
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class To_KV(nn.Module):
    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        encoder_hid_proj: Optional[Mapping[str, torch.Tensor]] = None,
        weight_kolors=1.0,
    ):
        super().__init__()

        if encoder_hid_proj is not None:
            hid_proj = nn.Linear(encoder_hid_proj["weight"].shape[1], encoder_hid_proj["weight"].shape[0], bias=True)
            hid_proj.weight.data = encoder_hid_proj["weight"] * weight_kolors
            hid_proj.bias.data = encoder_hid_proj["bias"] * weight_kolors
        else:
            hid_proj = nn.Identity()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            linear_proj = nn.Linear(value.shape[1], value.shape[0], bias=False)
            linear_proj.weight.data = value
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Sequential(hid_proj, linear_proj)


def FeedForward(dim: int, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self.reshape_tensor(q, self.heads)
        k = self.reshape_tensor(k, self.heads)
        v = self.reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

    @staticmethod
    def reshape_tensor(x: torch.Tensor, heads: int):
        bs, length, width = x.shape
        # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
        x = x.view(bs, length, heads, -1)
        # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
        x = x.transpose(1, 2)
        # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
        x = x.reshape(bs, heads, length, -1)
        return x


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = self.masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:  # type: ignore
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)

    @staticmethod
    def masked_mean(t: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None):
        if mask is None:
            return t.mean(dim=dim)

        denom = mask.sum(dim=dim, keepdim=True)
        mask = rearrange(mask, "b n -> b n 1")
        masked_t = t.masked_fill(~mask, 0.0)

        return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class FacePerceiverResampler(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for attn, ff in self.layers:  # type: ignore
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            nn.GELU(),
            nn.Linear(clip_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class MLPProjModelFaceId(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class ProjModelFaceIdPlus(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds: torch.Tensor, clip_embeds: torch.Tensor, scale=1.0, shortcut=False) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        embeds = image_embeds
        x = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class IPAdapter(nn.Module):
    def __init__(
        self,
        ipadapter_model,
        cross_attention_dim=1024,
        output_cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
        is_sdxl=False,
        is_plus=False,
        is_full=False,
        is_faceid=False,
        is_portrait_unnorm=False,
        is_kwai_kolors=False,
        encoder_hid_proj=None,
        weight_kolors=1.0,
    ):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_portrait_unnorm = is_portrait_unnorm
        self.is_kwai_kolors = is_kwai_kolors

        if is_faceid and not is_portrait_unnorm:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus or is_portrait_unnorm:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(
            ipadapter_model["ip_adapter"], encoder_hid_proj=encoder_hid_proj, weight_kolors=weight_kolors
        )

        self.multigpu_clones = {}

    def create_multigpu_clone(self, device):
        if device not in self.multigpu_clones:
            orig_multigpu_clones = self.multigpu_clones
            try:
                self.multigpu_clones = {}
                new_clone = copy.deepcopy(self)
                new_clone = new_clone.to(device)
                orig_multigpu_clones[device] = new_clone
            finally:
                self.multigpu_clones = orig_multigpu_clones

    def get_multigpu_clone(self, device):
        return self.multigpu_clones.get(device, self)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
        )
        return image_proj_model

    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl and not self.is_kwai_kolors else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim, clip_embeddings_dim=self.clip_embeddings_dim
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=self.clip_extra_context_tokens,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed, batch_size):
        torch_device = comfy.model_management.get_torch_device()
        intermediate_device = comfy.model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        clip_embed = torch.split(clip_embed, batch_size, dim=0)
        clip_embed_zeroed = torch.split(clip_embed_zeroed, batch_size, dim=0)

        image_prompt_embeds = []
        uncond_image_prompt_embeds = []

        for ce, cez in zip(clip_embed, clip_embed_zeroed):
            image_prompt_embeds.append(self.image_proj_model(ce.to(torch_device)).to(intermediate_device))
            uncond_image_prompt_embeds.append(self.image_proj_model(cez.to(torch_device)).to(intermediate_device))

        del clip_embed, clip_embed_zeroed

        image_prompt_embeds = torch.cat(image_prompt_embeds, dim=0)
        uncond_image_prompt_embeds = torch.cat(uncond_image_prompt_embeds, dim=0)

        torch.cuda.empty_cache()

        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut, batch_size):
        torch_device = comfy.model_management.get_torch_device()
        intermediate_device = comfy.model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        face_embed_batch = torch.split(face_embed, batch_size, dim=0)
        clip_embed_batch = torch.split(clip_embed, batch_size, dim=0)

        embeds = []

        for face_embed, clip_embed in zip(face_embed_batch, clip_embed_batch):
            embeds.append(
                self.image_proj_model(
                    face_embed.to(torch_device), clip_embed.to(torch_device), scale=s_scale, shortcut=shortcut
                ).to(intermediate_device)
            )

        del face_embed_batch, clip_embed_batch

        embeds = torch.cat(embeds, dim=0)

        torch.cuda.empty_cache()

        return embeds
