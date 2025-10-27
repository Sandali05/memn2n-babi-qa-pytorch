#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Memory Network (MemN2N) for Facebook bAbI tasks.

Features:
- Multi-hop attention over memories
- Position Encoding (Sukhbaatar et al., 2015)
- Optional Temporal Encoding (trainable) per memory slot
- Adjacent weight tying across hops (A_k+1 = C_k)
- Padding-safe embeddings (padding_idx=0)
- Clean data pipeline for bAbI v1.2 (en / en-10k)

Author: (you)
License: MIT
"""

import argparse
import collections
import io
import math
import os
import random
import re
import tarfile
import urllib.request
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ----------------------------
# Utilities & Reproducibility
# ----------------------------

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# bAbI dataset download/paths
# ----------------------------

BABI_URLS = [
    # Keras example S3 mirror (commonly used)
    "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
    # Facebook archive mirror on GitHub
    "https://github.com/facebookarchive/bAbI-tasks/raw/master/tasks_1-20_v1-2.tar.gz",
]

TASK_NAME_BY_ID = {
    1: "single-supporting-fact",
    2: "two-supporting-facts",
    3: "three-supporting-facts",
    4: "two-arg-relations",
    5: "three-arg-relations",
    6: "yes-no-questions",
    7: "counting",
    8: "lists-sets",
    9: "simple-negation",
    10: "indefinite-knowledge",
    11: "basic-coreference",
    12: "conjunction",
    13: "compound-coreference",
    14: "time-reasoning",
    15: "basic-deduction",
    16: "basic-induction",
    17: "positional-reasoning",
    18: "size-reasoning",
    19: "path-finding",
    20: "agents-motivations",
}


def maybe_download_and_extract(root_dir: str) -> str:
    """
    Download and extract bAbI v1-2 if not already present.
    Returns the directory where 'tasks_1-20_v1-2' lives.
    """
    os.makedirs(root_dir, exist_ok=True)
    target_dir = os.path.join(root_dir, "tasks_1-20_v1-2")
    if os.path.isdir(target_dir):
        return target_dir

    archive_path = os.path.join(root_dir, "tasks_1-20_v1-2.tar.gz")
    if not os.path.exists(archive_path):
        print(f"[info] Downloading bAbI dataset to {archive_path} ...")
        last_err = None
        for url in BABI_URLS:
            try:
                urllib.request.urlretrieve(url, archive_path)
                print(f"[info] Downloaded from: {url}")
                break
            except Exception as e:
                last_err = e
                print(f"[warn] Failed to download from {url}: {e}")
        if not os.path.exists(archive_path):
            raise RuntimeError(f"Could not download bAbI archive. Last error: {last_err}")

    print(f"[info] Extracting {archive_path} ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=root_dir)
    if not os.path.isdir(target_dir):
        # Some mirrors nest differently; search for it
        for root, dirs, _ in os.walk(root_dir):
            if os.path.basename(root) == "tasks_1-20_v1-2":
                return root
        raise RuntimeError("Extracted, but could not locate 'tasks_1-20_v1-2' directory.")
    return target_dir


def get_task_files(base_dir: str, task_id: int, babi_10k: bool) -> Tuple[str, str]:
    """
    Given the root 'tasks_1-20_v1-2', return (train_file, test_file) for a task.
    """
    if task_id not in TASK_NAME_BY_ID:
        raise ValueError(f"Invalid task_id {task_id}. Must be in 1..20.")
    variant = "en-10k" if babi_10k else "en"
    tname = TASK_NAME_BY_ID[task_id]
    subdir = os.path.join(base_dir, variant)
    train = os.path.join(subdir, f"qa{task_id}_{tname}_train.txt")
    test = os.path.join(subdir, f"qa{task_id}_{tname}_test.txt")
    if not os.path.isfile(train) or not os.path.isfile(test):
        # Some mirrors store under variant without underscore
        # Fallback: search for files
        candidates = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.startswith(f"qa{task_id}_{tname}_train") and f.endswith(".txt"):
                    candidates.append(os.path.join(root, f))
        if candidates:
            # Guess siblings for test
            train = sorted([p for p in candidates if "_train" in os.path.basename(p)])[0]
            test = train.replace("_train", "_test")
    if not os.path.isfile(train) or not os.path.isfile(test):
        raise FileNotFoundError(f"Could not find train/test for task {task_id} in {base_dir}")
    return train, test


# ----------------------------
# Parsing & Vectorization
# ----------------------------

_PUNCT_RE = re.compile(r"([.,!?;])")


def tokenize(sentence: str) -> List[str]:
    """Split a sentence into tokens, preserving .,!?; as separate tokens."""
    sentence = sentence.strip()
    sentence = _PUNCT_RE.sub(r" \1 ", sentence)
    return [w.lower() for w in sentence.split()]


def parse_stories(lines: List[str], only_supporting: bool = False):
    """
    Parse bAbI stories.
    Each story is a list of sentences; each entry is (story, question, answer).
    If only_supporting=True, keep only sentences that support the answer (using provided indices).
    """
    data = []
    story = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        nid, line = raw.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []

        if '\t' in line:
            # question line
            q, a, supporting = line.split('\t')
            q_tokens = tokenize(q)
            supporting_ids = list(map(int, supporting.split())) if supporting.strip() else []
            if only_supporting:
                substory = [sent for idx, sent in story if idx in supporting_ids]
            else:
                substory = [sent for idx, sent in story]
            data.append((substory, q_tokens, a.lower()))
            story.append((nid, []))  # place holder to keep continuity
        else:
            # regular sentence
            sent_tokens = tokenize(line)
            story.append((nid, sent_tokens))
    return data


def get_stories(path: str, only_supporting: bool = False):
    with io.open(path, 'r', encoding='utf-8') as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def build_vocab(train_data, test_data):
    """
    Build word->id (vocab) including answers from both splits.
    Index 0 is reserved for PAD.
    """
    vocab = set()
    for (S, q, a) in train_data + test_data:
        for sent in S:
            vocab |= set(sent)
        vocab |= set(q)
        vocab.add(a)
    # Sorted for stable ids
    vocab = sorted(vocab)
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # 1..V
    word2idx["__pad__"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def vectorize_data(data, word2idx: Dict[str, int], memory_size: int,
                   sentence_len: int, query_len: int):
    """
    Convert token lists into fixed-size integer arrays.
    - Stories => shape [N, memory_size, sentence_len]
    - Questions => shape [N, query_len]
    - Answers => shape [N] integer class ids
    """
    S_array = []
    Q_array = []
    A_array = []

    for story, q, a in data:
        # Keep only the last 'memory_size' sentences, as is common in MemN2N
        story = story[-memory_size:]

        # Vectorize sentences
        mem = []
        for sent in story:
            ids = [word2idx.get(w, 0) for w in sent][:sentence_len]
            ids = ids + [0] * (sentence_len - len(ids))
            mem.append(ids)
        # Pad memory to fixed number of slots
        if len(mem) < memory_size:
            mem = [[0] * sentence_len] * (memory_size - len(mem)) + mem  # left pad so newer sentences at the end

        # Vectorize question
        q_ids = [word2idx.get(w, 0) for w in q][:query_len]
        q_ids = q_ids + [0] * (query_len - len(q_ids))

        # Answer id
        a_id = word2idx.get(a, 0)

        S_array.append(mem)
        Q_array.append(q_ids)
        A_array.append(a_id)

    return np.array(S_array, dtype=np.int64), np.array(Q_array, dtype=np.int64), np.array(A_array, dtype=np.int64)


# ----------------------------
# Position & Temporal Encoding
# ----------------------------

def position_encoding(sentence_len: int, embed_dim: int) -> torch.Tensor:
    """
    Create the position encoding matrix L in Sukhbaatar et al.:
    L[j,k] = (1 - j/J) - (k/d)*(1 - 2j/J)
    where j=1..J (word position), k=1..d (embedding dim)
    """
    J = sentence_len
    d = embed_dim
    L = np.zeros((J, d), dtype=np.float32)
    for j in range(1, J + 1):
        for k in range(1, d + 1):
            L[j - 1, k - 1] = (1.0 - j / J) - (k / d) * (1.0 - 2.0 * j / J)
    return torch.tensor(L)  # shape [J, d]


# ----------------------------
# PyTorch Dataset
# ----------------------------

class BabiDataset(Dataset):
    def __init__(self, stories: np.ndarray, questions: np.ndarray, answers: np.ndarray):
        self.S = stories
        self.Q = questions
        self.A = answers

    def __len__(self):
        return self.S.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.S[idx]),  # [M, S]
            torch.from_numpy(self.Q[idx]),  # [Q]
            torch.tensor(self.A[idx], dtype=torch.long),  # []
        )


# ----------------------------
# MemN2N Model
# ----------------------------

class MemN2N(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 sentence_len: int,
                 query_len: int,
                 memory_size: int,
                 hops: int = 3,
                 dropout: float = 0.1,
                 use_time: bool = True,
                 padding_idx: int = 0):
        """
        End-to-end Memory Network with:
          - Position Encoding for sentences and questions
          - Optional Temporal Encoding per memory slot
          - Adjacent weight tying: A_{k+1} = C_k
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sentence_len = sentence_len
        self.query_len = query_len
        self.memory_size = memory_size
        self.hops = hops
        self.use_time = use_time

        # Embeddings for memory (A) and output (C) per hop, with adjacent tying
        self.A = nn.ModuleList()
        self.C = nn.ModuleList()

        # First hop A1 and C1
        self.A.append(nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx))
        self.C.append(nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx))

        for _ in range(1, hops):
            # Adjacent tying: A[k] shall share weights with C[k-1]
            A_k = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            C_prev = self.C[-1]
            # Tie: point A_k weight to C_prev weight (shared)
            A_k.weight = C_prev.weight
            self.A.append(A_k)

            C_k = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            self.C.append(C_k)

        # Query embedding B (separate from A/C)
        self.B = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Optional temporal encoding per memory slot for A and C
        if use_time:
            self.TA = nn.Parameter(torch.zeros(memory_size, embed_dim))
            self.TC = nn.Parameter(torch.zeros(memory_size, embed_dim))
            nn.init.normal_(self.TA, std=0.1)
            nn.init.normal_(self.TC, std=0.1)
        else:
            self.register_parameter("TA", None)
            self.register_parameter("TC", None)

        # Output projection. Often tied to embedding; here keep it free & stable.
        self.classifier = nn.Linear(embed_dim, vocab_size, bias=True)

        # Dropout (applied on u^k after each hop)
        self.dropout = nn.Dropout(p=dropout)

        # Position encodings (registered buffers so they move with .to(device))
        self.register_buffer("PE_sent", position_encoding(sentence_len, embed_dim))  # [S, d]
        self.register_buffer("PE_query", position_encoding(query_len, embed_dim))    # [Q, d]

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for embeddings and linear layer (except padding_idx rows are zeroed)
        for emb_list in [self.A, self.C, [self.B]]:
            for emb in emb_list:
                if emb is None:
                    continue
                nn.init.xavier_uniform_(emb.weight)
                if emb.padding_idx is not None:
                    with torch.no_grad():
                        emb.weight[emb.padding_idx].fill_(0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _embed_sentences(self, E: nn.Embedding, story: torch.Tensor) -> torch.Tensor:
        """
        story: [B, M, S] token ids
        returns memory embeddings: [B, M, d] with position encoding summed over words
        """
        B, M, S = story.shape
        # [B, M, S, d]
        e = E(story)
        # position-encode: elementwise multiply by PE_sent [S, d], then sum over S
        # broadcasting: [B, M, S, d] * [S, d] -> [B, M, S, d]
        e = e * self.PE_sent
        m = e.sum(dim=2)  # [B, M, d]
        return m

    def _embed_query(self, q_ids: torch.Tensor) -> torch.Tensor:
        """
        q_ids: [B, Q]
        return u^1: [B, d]
        """
        e = self.B(q_ids)                           # [B, Q, d]
        e = e * self.PE_query                       # [B, Q, d]
        u = e.sum(dim=1)                            # [B, d]
        return u

    def forward(self, story: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """
        story: [B, M, S]
        question: [B, Q]
        returns logits over vocab: [B, V]
        """
        device = story.device
        B, M, S = story.shape

        # mask for empty memory rows: True where memory slot is PAD-only
        # A slot is PAD-only if sum over words == 0
        with torch.no_grad():
            mem_nonpad = (story != 0).any(dim=2)  # [B, M]
        neg_inf = torch.tensor(-1e9, device=device)

        u = self._embed_query(question)  # [B, d]
        u = self.dropout(u)

        for k in range(self.hops):
            m = self._embed_sentences(self.A[k], story)  # [B, M, d]
            c = self._embed_sentences(self.C[k], story)  # [B, M, d]

            if self.use_time:
                # Add temporal encoding per memory slot index
                m = m + self.TA.unsqueeze(0)  # [1, M, d] -> [B, M, d]
                c = c + self.TC.unsqueeze(0)  # [1, M, d] -> [B, M, d]

            # Attention
            # p_i = softmax(u dot m_i)
            # [B, M]
            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2)

            # Mask out padded memory rows so they don't get attention
            scores = scores.masked_fill(~mem_nonpad, neg_inf)
            p = F.softmax(scores, dim=1)  # [B, M]

            # Output vector
            # o = sum_i p_i * c_i
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)  # [B, d]

            # Update u (adjacent tying uses identity)
            u = self.dropout(u + o)

        logits = self.classifier(u)  # [B, V]
        return logits


# ----------------------------
# Training / Evaluation
# ----------------------------

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 40.0):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for S, Q, A in tqdm(loader, desc="train", leave=False):
        S, Q, A = S.to(device), Q.to(device), A.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(S, Q)
        loss = criterion(logits, A)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = A.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, A) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for S, Q, A in tqdm(loader, desc="eval", leave=False):
        S, Q, A = S.to(device), Q.to(device), A.to(device)
        logits = model(S, Q)
        loss = criterion(logits, A)
        bs = A.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, A) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def qualitative_examples(model, loader, idx2word, k: int = 5):
    """
    Print K random predictions for sanity check.
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect a small batch
    for S, Q, A in loader:
        S, Q, A = S.to(device), Q.to(device), A.to(device)
        logits = model(S, Q)
        preds = logits.argmax(dim=-1)
        # decode and print
        for i in range(min(k, S.size(0))):
            gold = idx2word.get(A[i].item(), "<unk>")
            pred = idx2word.get(preds[i].item(), "<unk>")
            q_tokens = [idx2word.get(t.item(), "") for t in Q[i] if t.item() != 0]
            # retrieve the last non-empty sentence in memory
            mem_tokens = []
            for sent in S[i].tolist():
                toks = [idx2word.get(t, "") for t in sent if t != 0]
                if toks:
                    mem_tokens.append(" ".join(toks))
            print("-" * 60)
            print("Q:", " ".join(q_tokens))
            print("Pred:", pred, "| Gold:", gold)
            if mem_tokens:
                print("Mem (last 3):")
                for snt in mem_tokens:
                    print("  •", snt)
        break


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="MemN2N for bAbI QA")
    parser.add_argument("--data_dir", type=str, default="./data/babi", help="Where to store/download bAbI")
    parser.add_argument("--task_id", type=int, default=1, help="Task id in 1..20")
    parser.add_argument("--babi_10k", action="store_true", help="Use en-10k (10k training samples) variant")
    parser.add_argument("--only_supporting", action="store_true", help="Train on only supporting sentences")
    parser.add_argument("--memory_size", type=int, default=50, help="Max # of memory slots (sentences)")
    parser.add_argument("--sentence_len", type=int, default=20, help="Max words per sentence")
    parser.add_argument("--query_len", type=int, default=20, help="Max words per question")
    parser.add_argument("--embed_dim", type=int, default=50, help="Embedding / hidden size")
    parser.add_argument("--hops", type=int, default=3, help="# of memory hops")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout on controller state")
    parser.add_argument("--use_time", action="store_true", help="Enable temporal encoding per memory slot")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=40.0)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of train used for validation")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[info] Using device: {device}")

    # Prepare data
    root = maybe_download_and_extract(args.data_dir)
    train_file, test_file = get_task_files(root, args.task_id, babi_10k=args.babi_10k)

    print(f"[info] Loading task {args.task_id}: {os.path.basename(train_file)}")
    train_raw = get_stories(train_file, only_supporting=args.only_supporting)
    test_raw = get_stories(test_file, only_supporting=args.only_supporting)

    # Infer maxima from data if user left defaults too small
    max_sent_len = max([len(s) for S, _, _ in train_raw + test_raw for s in S] + [args.sentence_len])
    max_query_len = max([len(q) for _, q, _ in train_raw + test_raw] + [args.query_len])

    sentence_len = max(args.sentence_len, max_sent_len)
    query_len = max(args.query_len, max_query_len)
    memory_size = args.memory_size

    word2idx, idx2word = build_vocab(train_raw, test_raw)
    vocab_size = len(word2idx)

    print(f"[info] Vocab size: {vocab_size} | sentence_len={sentence_len} | query_len={query_len} | memory_size={memory_size}")

    # Vectorize
    S_train, Q_train, A_train = vectorize_data(train_raw, word2idx, memory_size, sentence_len, query_len)
    S_test, Q_test, A_test = vectorize_data(test_raw, word2idx, memory_size, sentence_len, query_len)

    # Split train into train/val
    full_train_ds = BabiDataset(S_train, Q_train, A_train)
    val_size = int(len(full_train_ds) * args.val_split)
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    test_ds = BabiDataset(S_test, Q_test, A_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model
    model = MemN2N(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        sentence_len=sentence_len,
        query_len=query_len,
        memory_size=memory_size,
        hops=args.hops,
        dropout=args.dropout,
        use_time=args.use_time,
        padding_idx=0,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={te_loss:.4f} acc={te_acc:.4f}")

    # Qualitative preview
    print("\nSome qualitative predictions:")
    qualitative_examples(model, test_loader, idx2word, k=5)


if __name__ == "__main__":
    main()
