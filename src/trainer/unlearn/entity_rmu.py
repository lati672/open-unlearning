# entity_rmu.py  (left-padding optimized)
import torch
from typing import List, Sequence, Tuple, Optional
from rmu import RMU  # your existing RMU class

# -------- helpers --------
def _find_subseq_positions(haystack: Sequence[int], needle: Sequence[int]) -> List[Tuple[int, int]]:
    """All (start, end_exclusive) matches of needle in haystack (non-overlapping)."""
    hits = []
    n = len(needle)
    if n == 0 or n > len(haystack):
        return hits
    i = 0
    hay = list(haystack)
    ned = list(needle)
    while True:
        try:
            i = hay.index(ned[0], i)
        except ValueError:
            break
        if hay[i:i+n] == ned:
            hits.append((i, i+n))
            i += n
        else:
            i += 1
    return hits

def _build_entity_mask_from_ids_leftpad(
    batch_input_ids: torch.Tensor,            # (B, S) long
    batch_attention_mask: torch.Tensor,       # (B, S) 1 for real tokens, 0 for pads (left-padded)
    batch_entity_ids: Optional[List[List[int]]],  # len B, each: entity token-id seq
    allow_possessive: bool = True,
    possessive_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Left-padding optimized: only search within the *content* slice given by attention_mask.
    Returns (B, S) bool mask with True at entity token positions.
    """
    B, S = batch_input_ids.shape
    mask = torch.zeros((B, S), dtype=torch.bool, device=batch_input_ids.device)
    if not batch_entity_ids:
        return mask

    # Precompute effective lengths (number of non-pad tokens)
    # Works for left padding because real tokens are right-aligned.
    lengths = batch_attention_mask.sum(dim=1).tolist()  # list of ints

    for b in range(B):
        L = int(lengths[b])
        if L <= 0:
            continue  # all pads
        start = S - L            # first real-token index
        end = S                  # exclusive
        ids_slice = batch_input_ids[b, start:end].tolist()

        entity = batch_entity_ids[b] if batch_entity_ids[b] is not None else []
        needles: List[List[int]] = []
        if len(entity) > 0:
            needles.append(list(entity))
            if allow_possessive and possessive_ids:
                needles.append(list(entity) + list(possessive_ids))

        for nd in needles:
            for s_rel, e_rel in _find_subseq_positions(ids_slice, nd):
                s_abs = start + s_rel
                e_abs = start + e_rel
                if 0 <= s_abs < e_abs <= S:
                    mask[b, s_abs:e_abs] = True
    return mask

def _align_entity_ids(default_ids: List[List[int]], B: int) -> List[List[int]]:
    """Repeat/truncate default list to length B."""
    if B <= 0:
        return []
    if not default_ids:
        return [[] for _ in range(B)]
    out, i = [], 0
    while len(out) < B:
        out.append(list(default_ids[i % len(default_ids)]))
        i += 1
    return out[:B]

# -------- EntityRMU (left-pad aware) --------
class EntityRMU(RMU):
    """
    RMU variant that applies unlearning ONLY to spans matching provided entity_token_ids.
    If the batch doesn't include them, we auto-inject defaults (Basil / Nikolai).
    Left-padding aware: uses attention_mask to search only the content slice.
    """

    def __init__(
        self,
        allow_possessive: bool = True,
        default_entity_token_ids_forget: Optional[List[List[int]]] = None,
        default_entity_token_ids_retain: Optional[List[List[int]]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.allow_possessive = allow_possessive

        # Hard-coded defaults from your examples
        basil = [33, 30149, 16566, 69, 109475, 1708, 16222, 84, 11748, 72]   # Basil Mahfouz Al-Kuwaiti
        nikolai = [45, 1609, 337, 2192, 3765, 321, 869]                      # Nikolai Abilov

        self.default_entity_token_ids_forget = (
            default_entity_token_ids_forget if default_entity_token_ids_forget is not None
            else [basil, nikolai]
        )
        self.default_entity_token_ids_retain = (
            default_entity_token_ids_retain if default_entity_token_ids_retain is not None
            else None  # use complement for retain by default
        )

        self._possessive_ids: Optional[List[int]] = None  # lazily filled

    def _lazy_init_possessive_ids(self):
        if self._possessive_ids is None and self.allow_possessive:
            tok = getattr(self, "tokenizer", None)
            if tok is not None:
                self._possessive_ids = tok("'s", add_special_tokens=False).get("input_ids", []) or []
            else:
                self._possessive_ids = []

    def _safe_mask(self, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        has_true = mask.any(dim=1)
        if has_true.all():
            return mask
        fixed = mask.clone()
        idx = (~has_true).nonzero(as_tuple=False).flatten()
        if idx.numel() > 0:
            fixed[idx] = fallback[idx]
        return fixed

    # -------------- main loss --------------
    def compute_loss(self, model, inputs, return_outputs=False):
        # -------- forget --------
        f = inputs["forget"]
        forget_inputs = {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "labels": f["labels"],
        }

        model_forget_act, forget_outputs = self._forward_with_cache(
            model, forget_inputs, module=self.model_module, no_grad=False
        )

        ctrl_vec = f.get("control_vec", None)
        if ctrl_vec is None:
            ctrl_vec = self.get_control_vector(
                dim=model_forget_act.shape[-1],
                device=model_forget_act.device,
                dtype=model_forget_act.dtype,
            )
        else:
            ctrl_vec = ctrl_vec.to(device=model_forget_act.device, dtype=model_forget_act.dtype)
        ctrl_vec = ctrl_vec.expand_as(model_forget_act)

        labels_mask_forget = forget_inputs["labels"] != -100
        self._lazy_init_possessive_ids()

        ent_ids_forget: Optional[List[List[int]]] = f.get("entity_token_ids", None)
        if ent_ids_forget is None:
            B = forget_inputs["input_ids"].shape[0]
            ent_ids_forget = _align_entity_ids(self.default_entity_token_ids_forget, B)

        ent_mask_forget = _build_entity_mask_from_ids_leftpad(
            batch_input_ids=forget_inputs["input_ids"],
            batch_attention_mask=forget_inputs["attention_mask"],
            batch_entity_ids=ent_ids_forget,
            allow_possessive=self.allow_possessive,
            possessive_ids=self._possessive_ids,
        )
        ent_mask_forget = self._safe_mask(ent_mask_forget, labels_mask_forget)

        forget_loss = self.compute_activation_loss(model_forget_act, ctrl_vec, ent_mask_forget)

        # -------- retain --------
        r = inputs["retain"]
        retain_inputs = {
            "input_ids": r["input_ids"],
            "attention_mask": r["attention_mask"],
            "labels": r["labels"],
        }

        retain_loss = self._compute_retain_loss_entity(model, retain_inputs, r)

        # -------- combine --------
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def _compute_retain_loss_entity(self, model, retain_inputs: dict, raw_inputs: dict) -> torch.Tensor:
        # If not activation-matching, defer to parent
        if getattr(self, "retain_loss_type", None) != "EMBED_DIFF":
            return super().compute_retain_loss(model, retain_inputs)

        model_act, _ = self._forward_with_cache(model, retain_inputs, module=self.model_module, no_grad=False)
        ref_act, _ = self._forward_with_cache_nograd(self.ref_model, retain_inputs, module=self.ref_model)

        labels_mask = retain_inputs["labels"] != -100
        self._lazy_init_possessive_ids()

        ent_ids_retain: Optional[List[List[int]]] = raw_inputs.get("entity_token_ids", None)
        if ent_ids_retain is None and self.default_entity_token_ids_retain is not None:
            B = retain_inputs["input_ids"].shape[0]
            ent_ids_retain = _align_entity_ids(self.default_entity_token_ids_retain, B)

        if ent_ids_retain is None:
            ent_mask = torch.zeros_like(labels_mask, dtype=torch.bool)
        else:
            ent_mask = _build_entity_mask_from_ids_leftpad(
                batch_input_ids=retain_inputs["input_ids"],
                batch_attention_mask=retain_inputs["attention_mask"],
                batch_entity_ids=ent_ids_retain,
                allow_possessive=self.allow_possessive,
                possessive_ids=self._possessive_ids,
            )

        # Preserve everything except entity spans
        non_ent_mask = (~ent_mask) & labels_mask
        non_ent_mask = self._safe_mask(non_ent_mask, labels_mask)

        return self.compute_activation_loss(
            model_act, ref_act.to(model_act.device, model_act.dtype), non_ent_mask
        )
