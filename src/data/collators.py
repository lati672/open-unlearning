import torch
import transformers
from typing import Dict, List, Sequence, Any, Callable, Optional, Tuple
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class EntityMaskedSupervisedCollator:
    """
    New collator for supervised fine-tuning that:
      • Pads input_ids and labels (dynamic to the longest in the batch).
      • Builds attention_mask from padded input_ids.
      • Adds a padded boolean `entity_mask` (only for branches you choose, e.g. "forget").

    Data format per sample (nested is fine):
      {
        "forget": {
          "input_ids": 1D LongTensor/array-like (UNPADDED),
          "labels":    1D LongTensor/array-like (UNPADDED, with IGNORE_INDEX where needed),
          # EITHER provide one of:
          "entity_mask": 1D BoolTensor/array-like (UNPADDED),
          "text":        str  # if build_mask_fn is provided, collator will build mask
        },
        "retain": { ... }  # same as above, but no entity_mask needed unless you want it
      }

    Notes:
      • If neither precomputed `entity_mask` nor `build_mask_fn`+`text` is available,
        the collator will fall back to `labels != IGNORE_INDEX` for that branch.
      • Padding side: 'right' or 'left'.
      • `apply_to_branches` controls where we expect/add entity_mask (default: ["forget"]).
    """

    def __init__(
        self,
        tokenizer,
        padding_side: str = "right",
        apply_to_branches: Tuple[str, ...] = ("forget",),
        build_mask_fn: Optional[Callable[[List[str]], List[List[bool]]]] = None,
        text_key: str = "text",
        enforce_subset_of_labels: bool = True,
        index_field: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.padding_side = padding_side
        assert self.padding_side in ("right", "left")
        self.apply_to_branches = set(apply_to_branches)
        self.build_mask_fn = build_mask_fn
        self.text_key = text_key
        self.enforce_subset_of_labels = enforce_subset_of_labels
        self.index_field = index_field

    # ------------ public API ------------
    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return self._collate(instances, branch=None)

    # ------------ helpers ------------
    def _pad_1d(
        self,
        seqs: List[torch.Tensor],
        pad_value: Any,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        # Ensure tensors
        ts = [s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=dtype) for s in seqs]
        if dtype is not None:
            ts = [t.to(dtype) for t in ts]
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=pad_value)
        else:
            flipped = [torch.flip(t, dims=[0]) for t in ts]
            padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=pad_value)
            return torch.flip(padded, dims=[1])

    def _collate(self, items: Sequence[Dict[str, Any]], branch: Optional[str]) -> Dict[str, Any]:
        assert isinstance(items[0], dict)
        # Nested dict case (e.g., {"forget": {...}, "retain": {...}})
        if "input_ids" not in items[0]:
            out: Dict[str, Any] = {}
            for k in items[0].keys():
                sub_items = [it[k] for it in items]
                out[k] = self._collate(sub_items, branch=k)  # pass current branch name
            return out

        # Leaf case
        leaf = self._collate_leaf(items, branch)
        return leaf

    def _collate_leaf(self, items: Sequence[Dict[str, Any]], branch: Optional[str]) -> Dict[str, Any]:
        # ---- input_ids ----
        input_ids_list = [it["input_ids"] for it in items]
        input_ids = self._pad_1d(input_ids_list, self.pad_token_id, dtype=torch.long)

        # ---- attention_mask ----
        attention_mask = input_ids.ne(self.pad_token_id)

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # ---- labels (optional) ----
        if "labels" in items[0]:
            labels_list = [it["labels"] for it in items]
            labels = self._pad_1d(labels_list, IGNORE_INDEX, dtype=torch.long)
            out["labels"] = labels
        else:
            labels = None  # may be None; handle in entity_mask fallback

        # ---- entity_mask (only for selected branches) ----
        if branch in self.apply_to_branches:
            ent_masks_unpadded: Optional[List[List[bool]]] = None

            # (A) precomputed per-sample mask provided
            if "entity_mask" in items[0]:
                ent_masks_unpadded = [
                    (m.tolist() if isinstance(m, torch.Tensor) else m) for m in (it["entity_mask"] for it in items)
                ]

            # (B) or compute from text if build_mask_fn is provided
            elif self.build_mask_fn is not None and self.text_key in items[0]:
                texts = [it[self.text_key] for it in items]
                ent_masks_unpadded = self.build_mask_fn(texts)  # -> List[List[bool]]

            # (C) fallback: use labels-supervised positions if available, else zeros
            if ent_masks_unpadded is None:
                if labels is not None:
                    ent_unpadded = [ (torch.tensor(it["labels"]) != IGNORE_INDEX).tolist() for it in items ]
                else:
                    ent_unpadded = [ [False] * (it["input_ids"].shape[0] if isinstance(it["input_ids"], torch.Tensor) else len(it["input_ids"])) for it in items ]
                ent_masks_unpadded = ent_unpadded

            # pad entity mask with False
            ent_masks_padded = self._pad_1d(
                [torch.tensor(m, dtype=torch.bool) for m in ent_masks_unpadded],
                pad_value=False,
                dtype=torch.bool,
            )

            # optionally enforce entity ⊆ supervised labels
            if self.enforce_subset_of_labels and labels is not None:
                ent_masks_padded = ent_masks_padded & (labels != IGNORE_INDEX)

            out["entity_mask"] = ent_masks_padded

        # ---- optional index passthrough ----
        if self.index_field:
            if self.index_field in items[0]:
                out[self.index_field] = torch.tensor([it[self.index_field] for it in items], dtype=torch.long)

        return out
