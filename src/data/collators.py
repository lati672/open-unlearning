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


# --------------------------------- Collator (text-only matching) ---------------------------------
class DataCollatorWithEntityMask(object):
    """
    INPUT (per-sample): same as DataCollatorForSupervisedDataset
      Flat:
        {
          "input_ids": 1D LongTensor (unpadded),
          "labels":    1D LongTensor (optional, unpadded),
          "row_id":    int (optional),
          # optionally "entity_mask": 1D 0/1 list/tensor (unpadded)
        }
      Nested:
        { "forget": {...}, "retain": {...} }

    OUTPUT: identical keys as DataCollatorForSupervisedDataset (+ "entity_mask")
      Flat:
        {
          "input_ids":      (B, S) LongTensor,
          "attention_mask": (B, S) LongTensor,
          "labels":         (B, S) LongTensor  [if provided],
          "entity_mask":    (B, S) LongTensor, # 0/1
          "row_id":         (B,)   LongTensor  [if index="row_id" and present]
        }
      Nested:
        { "forget": {...}, "retain": {...} } (each branch has the flat structure above)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        author_names: List[str],
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.index = index

        # Ensure we can pad
        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

        assert padding_side in ("left", "right")
        self.padding_side = padding_side

        # Compile text-level regexes (full + subnames)
        self.rx_full, self.rx_subs = _build_regexes_for_names(author_names)

    # ---------- utilities (identical padding behavior) ----------
    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        return [instance[key] for instance in instances]

    def _pad_tokens(self, seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        else:
            return torch.nn.utils.rnn.pad_sequence(
                [torch.flip(x, dims=[0]) for x in seqs],
                batch_first=True,
                padding_value=pad_val,
            ).flip(dims=[1])

    # ---------- entity-mask (TEXT-ONLY) ----------
    def _build_entity_mask_via_text(self, ids: List[int]) -> List[int]:
        """
        Decode -> re-tokenize (no specials) -> regex on normalized text -> map spans via word_ids() if available;
        else via per-token offsets. If round-trip tokenization differs, return an all-zero mask (text-only policy).
        """
        if not ids:
            return []

        # 1) Decode without cleanup to keep spaces/newlines as-is (improves round-trip stability)
        text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # 2) Re-tokenize with offsets, no specials
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )

        ids_rt = enc["input_ids"]
        
        if list(ids_rt) != list(ids):
            # Text-only policy: do NOT fall back to ID matching; just return zeros (conservative).
            return [0] * len(ids)

        offsets = enc["offset_mapping"]
        word_ids = enc.word_ids() if hasattr(enc, "word_ids") else None
        text_norm = _nfkc_lower(text)

        # 3) Collect normalized spans (full names + subnames), then greedily make them non-overlapping
        spans: List[Tuple[int, int]] = []
        if self.rx_full:
            spans += [(m.start(), m.end()) for m in self.rx_full.finditer(text_norm)]
        if self.rx_subs:
            spans += [(m.start(), m.end()) for m in self.rx_subs.finditer(text_norm)]

        if not spans:
            return [0] * len(ids)

        spans.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
        chosen = []
        taken = [False] * (len(text_norm) + 1)
        for s, e in spans:
            if any(taken[s:e]):
                continue
            for i in range(s, e):
                taken[i] = True
            chosen.append((s, e))
        spans = chosen

        # 4) Map char spans -> token indices
        mask = [0] * len(ids)

        if word_ids is not None:
            # Mark all tokens whose word overlaps any span (so multi-piece BPEs are covered)
            matched_words = set()
            for ti, off in enumerate(offsets):
                if off is None:
                    continue
                s, e = off
                if s == e:
                    continue
                for (cs, ce) in spans:
                    if s < ce and cs < e:
                        wid = word_ids[ti]
                        if wid is not None:
                            matched_words.add(wid)
                        break
            for ti, wid in enumerate(word_ids):
                if wid is not None and wid in matched_words:
                    mask[ti] = 1
            return mask

        # Fallback (still text-only): no word_ids() → per-token overlap
        for ti, off in enumerate(offsets):
            if off is None:
                continue
            s, e = off
            if s == e:
                continue
            for (cs, ce) in spans:
                if s < ce and cs < e:
                    mask[ti] = 1
                    break

        return mask

    # ---------- main (mirrors DataCollatorForSupervisedDataset) ----------
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct: Dict[str, Any] = {}

        # NESTED CASE: recurse into keys until we hit leafs with input_ids
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(instances=instances, key=key)
                return_dct[key] = self(key_instances)
            return return_dct

        # LEAF CASE: pad & stack like the baseline collator
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
                return_dct.update({
                    self.index: torch.tensor([example[self.index] for example in instances])
                })
            else:
                raise Warning(f"{self.index} not found in dataset")

        # --- entity_mask (TEXT-ONLY) ---
        if "entity_mask" in instances[0]:
            masks = []
            for ex in instances:
                m = ex["entity_mask"]
                if isinstance(m, torch.Tensor):
                    m = m.tolist()
                masks.append(torch.tensor(m, dtype=torch.long))
            entity_mask = self._pad_tokens(masks, 0)
        else:
            masks = []
            for ex in instances:
                ids_list = ex["input_ids"].tolist()
                m = self._build_entity_mask_via_text(ids_list)  # text-only path
                masks.append(torch.tensor(m, dtype=torch.long))
            entity_mask = self._pad_tokens(masks, 0)

        return_dct["entity_mask"] = entity_mask
        return return_dct
