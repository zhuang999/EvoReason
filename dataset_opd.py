import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional

from dataset_codi import (
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
    LatentRecCoDiCollator,
    PRIMITIVE_TAGS,
    NUM_PRIMITIVES,
)

REFINE_PROMPT = "Please re-examine the user's purchase history and provide the correct recommendation."


class LatentRecOPDDataset(Dataset):
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.data = pd.read_parquet(data_path)
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data.head(max_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        return {
            "description": str(row["description"]),
            "groundtruth": str(row["groundtruth"]),
            "title": row.get("title", None),
            "categories": row.get("categories", None),
            "yr_cot": str(row.get("yr_cot", "")),
            "yr_sid": str(row.get("yr_sid", "")),
            "yo_sid": str(row.get("yo_sid", "")),
        }


class LatentRecOPDCollator:
    """
    TRD-style OPD collator. Produces fields for forward_opd():
      - encoder_input_ids / encoder_attention_mask: student encoder prefix
      - decoder_input_ids / labels: SID part (eot_id + SID + im_end)
      - yr_cot_ids / yr_cot_attention_mask: teacher-generated CoT text tokens
      - enc_lengths, cot_lengths, dec_lengths, model_answer_position
    """

    def __init__(self, tokenizer, max_length: int = 4096, max_cot_length: int = 512,
                 bot_id: int = None, eot_id: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
        if self.sid_begin_id == tokenizer.unk_token_id:
            raise ValueError("<|sid_begin|> not in tokenizer vocab")
        self._base_collator = LatentRecCoDiCollator(
            tokenizer=tokenizer, max_length=max_length,
            bot_id=bot_id, eot_id=eot_id,
        )

    def _find_sid_begin_pos(self, token_ids: List[int]) -> int:
        for i, tid in enumerate(token_ids):
            if tid == self.sid_begin_id:
                return i
        return 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        base_feats = [
            {
                "description": f["description"],
                "groundtruth": f["groundtruth"],
                "title": f.get("title"),
                "categories": f.get("categories"),
            }
            for f in features
        ]
        base_batch = self._base_collator(base_feats)

        decoder_input_ids_list = []
        labels_list = []
        model_answer_positions = []
        dec_lengths_list = []
        yr_cot_ids_list = []
        cot_lengths_list = []
        primitive_ranges_list = []

        refine_ids = self.tokenizer(REFINE_PROMPT, add_special_tokens=False)["input_ids"]
        refine_len = len(refine_ids)

        for feat in features:
            yr_cot = feat.get("yr_cot", "")
            yr_sid = feat.get("yr_sid", "")
            groundtruth = feat["groundtruth"]

            sid_text = yr_sid if yr_sid else groundtruth
            decoder_str = f"{sid_text}<|im_end|>\n"
            dec_ids = self.tokenizer(
                decoder_str,
                truncation=True,
                max_length=256,
                add_special_tokens=False,
            )["input_ids"]
            if self.eot_id is not None:
                dec_ids = [self.eot_id] + dec_ids

            dec_labels = dec_ids[:]
            if self.eot_id is not None:
                dec_labels[0] = IGNORE_INDEX

            sid_pos = self._find_sid_begin_pos(dec_ids)

            cot_ids = self.tokenizer(
                yr_cot,
                truncation=True,
                max_length=self.max_cot_length,
                add_special_tokens=False,
            )["input_ids"]

            decoder_input_ids_list.append(torch.tensor(dec_ids, dtype=torch.long))
            labels_list.append(torch.tensor(dec_labels, dtype=torch.long))
            model_answer_positions.append(sid_pos)
            dec_lengths_list.append(len(dec_ids))
            yr_cot_ids_list.append(torch.tensor(cot_ids, dtype=torch.long))
            cot_lengths_list.append(len(cot_ids))

            prim_ranges = [[-1, -1]] * NUM_PRIMITIVES
            if NUM_PRIMITIVES > 0 and yr_cot:
                cot_text = yr_cot
                prim_ranges_i = []
                for i, tag in enumerate(PRIMITIVE_TAGS):
                    char_pos = cot_text.find(tag)
                    if char_pos == -1:
                        prim_ranges_i.append([-1, -1])
                        continue
                    before_text = cot_text[:char_pos]
                    before_ids = self.tokenizer(before_text, add_special_tokens=False)["input_ids"]
                    tok_start = len(before_ids)
                    end_text = cot_text[:char_pos + len(tag)]
                    end_ids = self.tokenizer(end_text, add_special_tokens=False)["input_ids"]
                    tok_end = len(end_ids)
                    if i + 1 < NUM_PRIMITIVES:
                        next_tag = PRIMITIVE_TAGS[i + 1]
                        next_pos = cot_text.find(next_tag, char_pos + len(tag))
                        if next_pos != -1:
                            next_before = cot_text[:next_pos]
                            next_before_ids = self.tokenizer(next_before, add_special_tokens=False)["input_ids"]
                            tok_end = len(next_before_ids)
                    prim_ranges_i.append([tok_start, tok_end])
                prim_ranges = prim_ranges_i
            primitive_ranges_list.append(prim_ranges)

        pad_id = self.tokenizer.pad_token_id

        decoder_padded = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )
        yr_cot_padded = torch.nn.utils.rnn.pad_sequence(
            yr_cot_ids_list, batch_first=True, padding_value=pad_id
        )

        return dict(
            encoder_input_ids=base_batch["encoder_input_ids"],
            encoder_attention_mask=base_batch["encoder_attention_mask"],
            decoder_input_ids=decoder_padded,
            labels=labels_padded,
            yr_cot_ids=yr_cot_padded,
            yr_cot_attention_mask=yr_cot_padded.ne(pad_id).long(),
            model_answer_position=torch.tensor(model_answer_positions, dtype=torch.long),
            enc_lengths=base_batch["enc_lengths"],
            cot_lengths=torch.tensor(cot_lengths_list, dtype=torch.long),
            dec_lengths=torch.tensor(dec_lengths_list, dtype=torch.long),
            primitive_token_ranges=torch.tensor(primitive_ranges_list, dtype=torch.long),
        )
