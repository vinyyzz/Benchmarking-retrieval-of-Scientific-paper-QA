from importlib.metadata import version
from pathlib import Path
import re

import numpy as np

EMBEDDINGS = {
    "E1": ("sentence-transformers/all-MiniLM-L6-v2", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
    "E2": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
}
MODEL_FILES = ("*.json", "*.txt", "model.safetensors", "1_Pooling/config.json")
DEPENDENCIES = ("numpy", "torch", "transformers", "tokenizers", "sentence-transformers")


def embedding_dependencies():
    return {package: version(package) for package in DEPENDENCIES}


def model_path(model_id, root):
    from huggingface_hub import snapshot_download

    repo, revision = EMBEDDINGS[model_id]
    root = Path(root)
    local = root / f"{model_id}-{revision}"
    if local.is_dir():
        return local
    return Path(snapshot_download(repo, revision=revision, allow_patterns=MODEL_FILES))


def canonical_tokenizer(root):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path("E1", root), use_fast=True, local_files_only=True)


class EncodingError(ValueError):
    """The encoder could not represent an input under the frozen policy."""


class Encoder:
    def __init__(self, model_id, root):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self.model = SentenceTransformer(str(model_path(model_id, root)), device="cpu",
                                         local_files_only=True, trust_remote_code=False)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        for text in texts:
            length = len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            if length > self.model.max_seq_length:
                raise EncodingError(f"Input has {length} native tokens, maximum is {self.model.max_seq_length}")
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=False, prompt="",
                                    convert_to_numpy=True, normalize_embeddings=False)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise EncodingError("Encoder returned a zero or nonfinite vector")
        return vectors / norms[:, None]


def validate_revision(revision):
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
