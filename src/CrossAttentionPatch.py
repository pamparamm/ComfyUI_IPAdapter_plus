import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from comfy.ldm.modules.attention import attention_sage, optimized_attention

from .IPAdapter import IPAdapter
from .utils import tensor_to_size


class Attn2Replace:
    def __init__(self, callback: Callable, **kwargs):
        self.callbacks = [callback]
        self.kwargs = [kwargs]
        self.multigpu_kwargs = {}

    def get_multigpu_kwargs(self, device):
        return self.multigpu_kwargs.get(device, self.kwargs)

    def add(self, callback: Callable, **kwargs):
        self.callbacks.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if "sigmas" in extra_options else 999999999.9

        device_kwargs = self.get_multigpu_kwargs(q.device)

        for i, callback in enumerate(self.callbacks):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **device_kwargs[i])

        return out.to(dtype=dtype)

    def to(self, device, *args, **kwargs):
        if not isinstance(device, torch.device):
            return self
        # if casted versions already taken care of, do nothing
        if device in self.multigpu_kwargs:
            return self
        # ignore casts to CPU, as IPAdapter currently always sticks to load device(s)
        if device == "cpu" or device == torch.device("cpu"):
            return self
        # for now, assume cond must be present
        if self.kwargs[0]["cond"] is None:
            return self
        if self.kwargs[0]["cond"].device == device:
            return self

        new_kwargs = []
        for kwargs_dict in self.kwargs:
            new_kwargs.append(kwargs_dict.copy())

        for kwargs_dict in new_kwargs:
            for key, value in kwargs_dict.items():
                if key == "ipadapter":
                    value.create_multigpu_clone(device)
                elif type(kwargs_dict[key]) == torch.Tensor:
                    kwargs_dict[key] = value.to(device)

        self.multigpu_kwargs[device] = new_kwargs
        return self


def ipadapter_attention_factory(
    attn2_key: tuple[str, int, int] = None,
    module_key="",
    ipadapter: IPAdapter = None,
    weight=1.0,
    cond: Tensor = None,
    cond_alt=None,
    uncond: Tensor = None,
    weight_type="linear",
    mask: Tensor | None = None,
    sigma_start=0.0,
    sigma_end=1.0,
    unfold_batch=False,
    embeds_scaling="V only",
    latent: dict[str, Tensor] = None,
    device="cuda",
    **kwargs,
) -> Callable:
    def _identity_attention(*args, **kwargs):
        return 0

    ipadapter = ipadapter.get_multigpu_clone(device)

    epsilon = 0.0
    if optimized_attention == attention_sage:
        epsilon = 1e-5

    block_type, block_id, t_idx = attn2_key
    bs = latent["samples"].shape[0]

    layers = 11 if "101_to_k_ip" in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    if weight_type == "ease in":
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == "ease out":
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == "ease in-out":
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "reverse in-out":
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "weak input" and block_type == "input":
        weight = weight * 0.2
    elif weight_type == "weak middle" and block_type == "middle":
        weight = weight * 0.2
    elif weight_type == "weak output" and block_type == "output":
        weight = weight * 0.2
    elif weight_type == "strong middle" and (block_type == "input" or block_type == "output"):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return _identity_attention

        if weight_type == "style transfer precise":
            if layers == 11 and t_idx == 3:
                uncond = cond
                cond = cond * epsilon
            elif layers == 16 and (t_idx == 4 or t_idx == 5):
                uncond = cond
                cond = cond * epsilon
        elif weight_type == "composition precise":
            if layers == 11 and t_idx != 3:
                uncond = cond
                cond = cond * epsilon
            elif layers == 16 and (t_idx != 4 and t_idx != 5):
                uncond = cond
                cond = cond * epsilon

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if (isinstance(weight, torch.Tensor) and torch.all(weight == 0)) or weight == 0:
        return _identity_attention

    # if unfold_batch:
    #     cond = tensor_to_size(cond, 2)
    #     uncond = tensor_to_size(uncond, 2)

    #     k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
    #     k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
    #     v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
    #     v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    # else:

    k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(bs, 1, 1)
    k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(bs, 1, 1)
    v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(bs, 1, 1)
    v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(bs, 1, 1)

    # if embeds_scaling == "K+mean(V) w/ C penalty":
    #     scaling = float(ip_k.shape[2]) / 1280.0
    #     weight = weight * scaling
    #     ip_k = ip_k * weight
    #     ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
    #     ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
    #     del ip_v_mean
    # elif embeds_scaling == "K+V w/ C penalty":
    #     scaling = float(ip_k.shape[2]) / 1280.0
    #     weight = weight * scaling
    #     ip_k = ip_k * weight
    #     ip_v = ip_v * weight
    # elif embeds_scaling == "K+V":
    #     ip_k = ip_k * weight
    #     ip_v = ip_v * weight
    # else:
    #     mult_out = weight  # I'm doing this to get the same results as before

    def _ipadapter_attention(out: Tensor, q: Tensor, k: Tensor, v: Tensor, extra_options, **kwargs):
        dtype = q.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]

        ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])

        if weight is not None:
            out_ip = out_ip * weight

        if mask is not None:
            h, w = extra_options["activations_shape"][-2:]

            _mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
            _mask = tensor_to_size(_mask, bs)

            _mask = _mask.repeat(len(cond_or_uncond), 1, 1)
            _mask = _mask.view(_mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

            out_ip = out_ip * _mask

        # out = out + out_ip

        return out_ip.to(dtype=dtype)

    return _ipadapter_attention


def ipadapter_attention(
    out,
    q,
    k,
    v,
    extra_options,
    module_key="",
    ipadapter=None,
    weight=1.0,
    cond=None,
    cond_alt=None,
    uncond=None,
    weight_type="linear",
    mask=None,
    sigma_start=0.0,
    sigma_end=1.0,
    unfold_batch=False,
    embeds_scaling="V only",
    **kwargs,
):
    ipadapter = ipadapter.get_multigpu_clone(q.device)

    epsilon = 0.0
    if optimized_attention == attention_sage:
        epsilon = 1e-5
    dtype = q.dtype
    cond_or_uncond = extra_options["cond_or_uncond"]
    block_type = extra_options["block"][0]
    # block_id = extra_options["block"][1]
    t_idx = extra_options["transformer_index"]
    layers = 11 if "101_to_k_ip" in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    # extra options for AnimateDiff
    ad_params = extra_options["ad_params"] if "ad_params" in extra_options else None

    b = q.shape[0]
    seq_len = q.shape[1]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    if weight_type == "ease in":
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == "ease out":
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == "ease in-out":
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "reverse in-out":
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "weak input" and block_type == "input":
        weight = weight * 0.2
    elif weight_type == "weak middle" and block_type == "middle":
        weight = weight * 0.2
    elif weight_type == "weak output" and block_type == "output":
        weight = weight * 0.2
    elif weight_type == "strong middle" and (block_type == "input" or block_type == "output"):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return 0

        if weight_type == "style transfer precise":
            if layers == 11 and t_idx == 3:
                uncond = cond
                cond = cond * epsilon
            elif layers == 16 and (t_idx == 4 or t_idx == 5):
                uncond = cond
                cond = cond * epsilon
        elif weight_type == "composition precise":
            if layers == 11 and t_idx != 3:
                uncond = cond
                cond = cond * epsilon
            elif layers == 16 and (t_idx != 4 and t_idx != 5):
                uncond = cond
                cond = cond * epsilon

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if unfold_batch:
        # Check AnimateDiff context window
        if ad_params is not None and ad_params["sub_idxs"] is not None:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, ad_params["full_length"])
                weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1)  # repeat for cond and uncond
            elif weight == 0:
                return 0

            # if image length matches or exceeds full_length get sub_idx images
            if cond.shape[0] >= ad_params["full_length"]:
                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
            # otherwise get sub_idxs images
            else:
                cond = tensor_to_size(cond, ad_params["full_length"])
                uncond = tensor_to_size(uncond, ad_params["full_length"])
                cond = cond[ad_params["sub_idxs"]]
                uncond = uncond[ad_params["sub_idxs"]]
        else:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, batch_prompt)
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1)  # repeat for cond and uncond
            elif weight == 0:
                return 0

            cond = tensor_to_size(cond, batch_prompt)
            uncond = tensor_to_size(uncond, batch_prompt)

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    else:
        # TODO: should we always convert the weights to a tensor?
        if isinstance(weight, torch.Tensor):
            weight = tensor_to_size(weight, batch_prompt)
            if torch.all(weight == 0):
                return 0
            weight = weight.repeat(len(cond_or_uncond), 1, 1)  # repeat for cond and uncond
        elif weight == 0:
            return 0

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)

    if len(cond_or_uncond) == 3:  # TODO: cosxl, I need to check this
        ip_k = torch.cat([(k_cond, k_uncond, k_cond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond, v_cond)[i] for i in cond_or_uncond], dim=0)
    else:
        ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    if embeds_scaling == "K+mean(V) w/ C penalty":
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        del ip_v_mean
    elif embeds_scaling == "K+V w/ C penalty":
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    elif embeds_scaling == "K+V":
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    else:
        # ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        out_ip = out_ip * weight  # I'm doing this to get the same results as before

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        # check if using AnimateDiff and sliding context window
        if mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None:
            # if mask length matches or exceeds full_length, get sub_idx masks
            if mask.shape[0] >= ad_params["full_length"]:
                mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, ad_params["full_length"])
                mask = mask[ad_params["sub_idxs"]]
        else:
            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start : crop_start + seq_len, :]

        out_ip = out_ip * mask

    # out = out + out_ip

    return out_ip.to(dtype=dtype)
