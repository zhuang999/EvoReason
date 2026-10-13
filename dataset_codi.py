import json
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional

IGNORE_INDEX = -100

DEFAULT_PRIMITIVE_TAGS: List[str] = []

PRIMITIVE_TAGS = list(DEFAULT_PRIMITIVE_TAGS)
NUM_PRIMITIVES = len(PRIMITIVE_TAGS)
LATENT_PER_PRIMITIVE = 1

SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend the next possible "
    "purchase for users based on their purchase history. Please predict the most likely next "
    "product that the user will purchase based on the user's historical purchase information."
)

THINK_START = "🧐"
THINK_END = " Millennia"


def load_primitive_tags_from_cot(cot_path: str) -> List[str]:
    tags = []
    seen = set()
    with open(cot_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = obj.get('think_content', '')
            for m in re.finditer(r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]', content):
                name = m.group(1)
                if name not in seen:
                    tags.append(name)
                    seen.add(name)
            if len(tags) >= 10:
                break
    return [f"[{t}]" for t in tags] if tags else list(DEFAULT_PRIMITIVE_TAGS)


def configure_primitives(cot_path: Optional[str] = None):
    global PRIMITIVE_TAGS, NUM_PRIMITIVES
    if cot_path:
        PRIMITIVE_TAGS = load_primitive_tags_from_cot(cot_path)
    else:
        PRIMITIVE_TAGS = list(DEFAULT_PRIMITIVE_TAGS)
    NUM_PRIMITIVES = len(PRIMITIVE_TAGS)


def _insert_identifiers_in_cot(cot_text: str, identifier_token_strs: List[str]) -> str:
    """Insert identifier tokens after each primitive segment (process right-to-left to avoid position shifts)."""
    result = cot_text
    for i in range(len(PRIMITIVE_TAGS) - 1, -1, -1):
        if i >= len(identifier_token_strs):
            continue
        ident = identifier_token_strs[i]
        if i + 1 < len(PRIMITIVE_TAGS):
            next_tag = PRIMITIVE_TAGS[i + 1]
            insert_pos = result.rfind(next_tag)
            if insert_pos == -1:
                result = result.rstrip() + '\n' + ident + '\n'
            else:
                result = result[:insert_pos] + ident + '\n' + result[insert_pos:]
        else:
            result = result.rstrip() + '\n' + ident
    return result


def build_cot_segments(title: Optional[str], categories: Optional[str], user_memory: Optional[str] = None) -> str:
    if not PRIMITIVE_TAGS:
        return ""
    cats = categories if (categories and not (isinstance(categories, float) and pd.isna(categories))) else "related categories"
    item_name = title if (title and not (isinstance(title, float) and pd.isna(title))) else "the target item"
    mem_text = user_memory[:200] if user_memory else "no long-term profile available"
    return (
        f"The user repeatedly engages with {cats}, favoring quality and reliability with low exploration.\n"
        f"The {cats} segment has diverse products; mid-range options with good features dominate.\n"
        f"Profile: {mem_text} Previously purchased items similar to {item_name} with high compatibility.\n"
        f"Recommend {item_name} based on preference alignment and category match.\n"
    )


def build_cot_text(title: Optional[str], categories: Optional[str]) -> str:
    if title is not None and not (isinstance(title, float) and pd.isna(title)):
        return (
            f"Based on the user's purchase history, the user has shown interest in {categories}. "
            f"Considering the patterns in their preferences, the next likely item would be in a related category."
        )
    return "The user is likely to buy items in a related category based on their purchase history."


def build_full_text(description: str, cot_text: str, groundtruth: str, user_memory: Optional[str] = None) -> str:
    user_content = description
    if user_memory:
        user_content = f"[User Profile]: {user_memory[:200]}\n\n{description}"
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{THINK_START}\n{cot_text}\n{THINK_END}\n"
        f"{groundtruth}<|im_end|>\n"
    )


class LatentRecCoDiDataset(Dataset):
    def __init__(self, data_path: str, max_samples: Optional[int] = None,
                 memory_path: Optional[str] = None, cot_path: Optional[str] = None):
        self.data = pd.read_parquet(data_path)
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data.head(max_samples)
        self.memory: Dict[int, str] = {}
        if memory_path is not None:
            with open(memory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get('type') == 'user':
                        self.memory[int(obj['id'])] = obj['memory']
        self.cot_data: Dict[str, str] = {}
        if cot_path is not None:
            with open(cot_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    uid = obj.get('user_id')
                    think = obj.get('think_content', '')
                    if uid is not None and think:
                        self.cot_data[str(uid)] = think
            print(f"Loaded {len(self.cot_data)} LLM-generated CoT entries from {cot_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        uid = row.get("user_id", None)
        user_memory = None
        if uid is not None and self.memory:
            try:
                user_memory = self.memory.get(int(uid) - 1, None)
            except (ValueError, TypeError):
                pass
        llm_cot = None
        if uid is not None and self.cot_data and str(uid) in self.cot_data:
            llm_cot = self.cot_data[str(uid)]
        return {
            "description": str(row["description"]),
            "groundtruth": str(row["groundtruth"]),
            "title": row.get("title", None),
            "categories": row.get("categories", None),
            "user_memory": user_memory,
            "llm_cot": llm_cot,
        }


class LatentRecCoDiCollator:

    def __init__(self, tokenizer, max_length: int = 4096, use_primitive_distill: bool = False,
                 identifier_token_ids: Optional[List[int]] = None,
                 bot_id: int = None, eot_id: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_primitive_distill = use_primitive_distill
        self.identifier_token_ids = identifier_token_ids or []
        self.use_identifier = bool(self.identifier_token_ids)
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
        if self.sid_begin_id == tokenizer.unk_token_id:
            raise ValueError("<|sid_begin|> not in tokenizer vocab")
        self.user_start_tokens = tokenizer.encode(
            "<|im_start|>user", add_special_tokens=False
        )
        if use_primitive_distill:
            self._primitive_tag_ids = [
                tokenizer.encode(tag, add_special_tokens=False)
                for tag in PRIMITIVE_TAGS
            ]

    def _find_subseq_pos(self, token_ids: List[int], subseq: List[int]) -> int:
        for i in range(len(token_ids) - len(subseq) + 1):
            if token_ids[i: i + len(subseq)] == subseq:
                return i
        return -1

    def _find_sid_begin_pos(self, token_ids: List[int]) -> int:
        for i, tid in enumerate(token_ids):
            if tid == self.sid_begin_id:
                return i
        raise ValueError("<|sid_begin|> not found in token sequence")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoder_input_ids_list = []
        decoder_input_ids_list = []
        ref_input_ids_list = []
        labels_list = []
        ref_labels_list = []
        ref_answer_positions = []
        model_answer_positions = []
        enc_lengths_list = []
        cot_lengths_list = []
        dec_lengths_list = []
        primitive_ranges_list = []
        identifier_positions_list = []

        for feat in features:
            description = feat["description"]
            groundtruth = feat["groundtruth"]
            title = feat.get("title")
            categories = feat.get("categories")
            user_memory = feat.get("user_memory")
            llm_cot = feat.get("llm_cot")

            if llm_cot:
                cot_text = llm_cot
            elif self.use_primitive_distill:
                cot_text = build_cot_segments(title, categories, user_memory=user_memory)
            else:
                cot_text = build_cot_text(title, categories)

            if self.use_identifier and self.identifier_token_ids:
                ident_strs = self.tokenizer.convert_ids_to_tokens(self.identifier_token_ids)
                cot_with_idents = _insert_identifiers_in_cot(cot_text, ident_strs)
            else:
                cot_with_idents = cot_text

            full_text = build_full_text(description, cot_with_idents, groundtruth, user_memory=user_memory)
            ref_enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_offsets_mapping=self.use_primitive_distill,
            )
            ref_ids = ref_enc["input_ids"]

            ref_labels = ref_ids[:]
            user_pos = self._find_subseq_pos(ref_ids, self.user_start_tokens)
            if user_pos != -1:
                ref_labels[:user_pos] = [IGNORE_INDEX] * user_pos
            else:
                ref_labels = [IGNORE_INDEX] * len(ref_labels)

            ref_answer_pos = self._find_sid_begin_pos(ref_ids)

            encoder_str = (
                f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                f"<|im_start|>user\n{description}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            enc_ids = self.tokenizer(
                encoder_str,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]
            if self.bot_id is not None:
                enc_ids = enc_ids + [self.bot_id]

            decoder_str = f"{groundtruth}<|im_end|>\n"
            dec_ids = self.tokenizer(
                decoder_str,
                truncation=True,
                max_length=256,
                add_special_tokens=False,
            )["input_ids"]
            if self.eot_id is not None:
                dec_ids = [self.eot_id] + dec_ids

            model_answer_pos = self._find_sid_begin_pos(dec_ids)

            labels = dec_ids[:]
            if self.eot_id is not None:
                labels[0] = IGNORE_INDEX

            cot_len = max(0, len(ref_ids) - len(enc_ids) - len(dec_ids))

            if self.use_primitive_distill:
                offsets = ref_enc.get("offset_mapping")
                prim_ranges = []
                tag_char_positions = []
                search_start = 0
                for tag in PRIMITIVE_TAGS:
                    pos = full_text.find(tag, search_start)
                    tag_char_positions.append(pos)
                    if pos != -1:
                        search_start = pos + len(tag)
                if offsets is not None:
                    for i in range(NUM_PRIMITIVES):
                        tag_char_start = tag_char_positions[i]
                        if tag_char_start == -1:
                            prim_ranges.append([-1, -1])
                            continue
                        tag_char_end = tag_char_start + len(PRIMITIVE_TAGS[i])
                        tok_start = -1
                        for t_idx, (c_s, c_e) in enumerate(offsets):
                            if c_s >= tag_char_end and c_e > c_s:
                                tok_start = t_idx
                                break
                        if i + 1 < NUM_PRIMITIVES and tag_char_positions[i + 1] != -1:
                            next_tag_char = tag_char_positions[i + 1]
                            tok_end = -1
                            for t_idx, (c_s, c_e) in enumerate(offsets):
                                if c_s >= next_tag_char and c_e > c_s:
                                    tok_end = t_idx
                                    break
                            if tok_end == -1:
                                tok_end = len(ref_ids) - len(dec_ids)
                        else:
                            tok_end = len(ref_ids) - len(dec_ids)
                        prim_ranges.append([tok_start if tok_start != -1 else -1, tok_end])
                else:
                    prim_ranges = [[-1, -1]] * NUM_PRIMITIVES
                primitive_ranges_list.append(prim_ranges)
            else:
                primitive_ranges_list.append([[-1, -1]] * NUM_PRIMITIVES)

            if self.use_identifier and self.identifier_token_ids:
                ident_pos = []
                for ident_id in self.identifier_token_ids:
                    pos = next((j for j, tid in enumerate(ref_ids) if tid == ident_id), -1)
                    ident_pos.append(pos)
                identifier_positions_list.append(ident_pos)
            else:
                identifier_positions_list.append([-1] * max(len(self.identifier_token_ids), 1))

            encoder_input_ids_list.append(torch.tensor(enc_ids, dtype=torch.long))
            decoder_input_ids_list.append(torch.tensor(dec_ids, dtype=torch.long))
            ref_input_ids_list.append(torch.tensor(ref_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            ref_labels_list.append(torch.tensor(ref_labels, dtype=torch.long))
            ref_answer_positions.append(ref_answer_pos)
            model_answer_positions.append(model_answer_pos)
            enc_lengths_list.append(len(enc_ids))
            cot_lengths_list.append(cot_len)
            dec_lengths_list.append(len(dec_ids))

        pad_id = self.tokenizer.pad_token_id

        encoder_padded = torch.nn.utils.rnn.pad_sequence(
            encoder_input_ids_list, batch_first=True, padding_value=pad_id
        )
        decoder_padded = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids_list, batch_first=True, padding_value=pad_id
        )
        ref_padded = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )
        ref_labels_padded = torch.nn.utils.rnn.pad_sequence(
            ref_labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            encoder_input_ids=encoder_padded,
            decoder_input_ids=decoder_padded,
            ref_input_ids=ref_padded,
            labels=labels_padded,
            ref_labels=ref_labels_padded,
            encoder_attention_mask=encoder_padded.ne(pad_id).long(),
            ref_attention_mask=ref_padded.ne(pad_id).long(),
            ref_answer_position=torch.tensor(ref_answer_positions, dtype=torch.long),
            model_answer_position=torch.tensor(model_answer_positions, dtype=torch.long),
            enc_lengths=torch.tensor(enc_lengths_list, dtype=torch.long),
            cot_lengths=torch.tensor(cot_lengths_list, dtype=torch.long),
            dec_lengths=torch.tensor(dec_lengths_list, dtype=torch.long),
            primitive_token_ranges=torch.tensor(primitive_ranges_list, dtype=torch.long),
            identifier_positions=torch.tensor(identifier_positions_list, dtype=torch.long),
        )
