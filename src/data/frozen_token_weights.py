"""Replay finite, normalized token weights for aligned supervised samples."""

import json

import torch

from trainer.masking import eligible_positions, sample_indices


class FrozenTokenWeightCollator:
    def __init__(self, collator, path):
        self.collator = collator
        self.records = {}
        with open(path) as stream:
            for line in stream:
                record = json.loads(line)
                index = record["doc_id"]
                weights = torch.tensor(record["weights"], dtype=torch.float32)
                size = len(record["input_ids"])
                if (type(index) is not int or index < 0 or index in self.records
                        or len(record["labels"]) != size or weights.shape != (size,)
                        or not torch.isfinite(weights).all() or (weights < 0).any()
                        or not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)):
                    raise ValueError(f"Invalid frozen token weight record for sample {index}")
                self.records[index] = (record["input_ids"], record["labels"], weights)

    def __call__(self, instances):
        batch = self.collator(instances)
        forget = batch.get("forget", batch)
        rows = [item.get("forget", item) for item in instances]
        weights = []
        for index, row in zip(sample_indices(forget), rows, strict=True):
            ids, labels, weight = self.records[index]
            if row["input_ids"].tolist() != ids or row["labels"].tolist() != labels:
                raise ValueError(f"Frozen token/label alignment differs for sample {index}")
            weights.append(weight)
        padded = self.collator._pad_tokens(weights, 0)
        eligible = eligible_positions(forget)
        if padded.shape != eligible.shape or (padded[~eligible] != 0).any():
            raise ValueError("Frozen token weights select non-eligible positions.")
        forget["token_weights"] = padded
        return batch
