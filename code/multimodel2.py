import os
import sys
import inspect
import copy

# ── Environment setup ───────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    _v = os.environ.get(_k, "")
    if "proxy_ip" in _v or "proxy_port" in _v:
        os.environ.pop(_k, None)

import json, math, random, shutil, logging, hashlib, re
from typing import List, Tuple, Dict, Optional, Any
from contextlib import nullcontext
from itertools import combinations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve, balanced_accuracy_score,
)
from sklearn.isotonic import IsotonicRegression
try:
    from sklearn.manifold import TSNE
    _HAVE_TSNE = True
except Exception:
    _HAVE_TSNE = False

from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pydicom; _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

try:
    import librosa; _HAVE_LIBROSA = True
except Exception:
    _HAVE_LIBROSA = False


# ═══════════════════════════════════════════════════════════════════════
# Runtime helpers
# ═══════════════════════════════════════════════════════════════════════
def _maybe_enable_tf32(enable: bool):
    if enable and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as AmpGradScaler
    _AMP_NEEDS_DEVICE_TYPE = True
except Exception:
    from torch.cuda.amp import autocast as _autocast        # type: ignore
    from torch.cuda.amp import GradScaler as AmpGradScaler  # type: ignore
    _AMP_NEEDS_DEVICE_TYPE = False


def amp_autocast(device: torch.device):
    if _AMP_NEEDS_DEVICE_TYPE:
        return _autocast(device_type=device.type,
                         enabled=(device.type == "cuda"))
    return _autocast(enabled=(device.type == "cuda"))


def safe_no_amp_context(device):
    dev_type = device.type if isinstance(device, torch.device) else "cpu"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=dev_type, enabled=False)
    if dev_type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


class RunningAverage:
    def __init__(self): self.avg = 0.0; self.count = 0

    def update(self, v: float):
        self.avg = (self.avg * self.count + float(v)) / (self.count + 1)
        self.count += 1

    def __call__(self): return self.avg


def set_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)03d:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def save_dict_to_json(d: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def save_checkpoint(state: dict, is_best: bool,
                    ckpt_dir: str, tag: str = ""):
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, f"last{tag}.pth.tar")
    torch.save(state, last)
    if is_best:
        shutil.copyfile(last,
                        os.path.join(ckpt_dir, f"best{tag}.pth.tar"))


def plot_metric(values: List[float], metric_name: str, save_path: str):
    plt.close("all")
    if not values:
        return
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, label=metric_name)
    plt.xlabel("Epoch"); plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_roc_pr(y_true, y_prob, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        return
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    else:
        fpr, tpr, auc = [0, 1], [0, 1], 0.0
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_roc.png")); plt.close()

    if len(np.unique(y_true)) > 1:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        prec, rec, ap = [1, 0], [0, 1], 0.0
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_pr.png")); plt.close()


def set_seed(seed: int = 72):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    strict_det = os.getenv("COCROSS_STRICT_DETERMINISM", "0") == "1"
    torch.backends.cudnn.benchmark = not strict_det
    torch.backends.cudnn.deterministic = strict_det
    try:
        torch.use_deterministic_algorithms(strict_det, warn_only=True)
    except Exception:
        pass


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(seed % (2 ** 32 - 1)); random.seed(seed)


set_seed(72)


# ═══════════════════════════════════════════════════════════════════════
# Offline text encoder (no config.json required)
# ═══════════════════════════════════════════════════════════════════════
class OfflineTextEncoder(nn.Module):
    """
    Lightweight Transformer text encoder built from scratch.
    Used when BiomedCLIP text files (config.json, vocab.txt) are missing.

    NOTE (methodology): on this dataset the clinical "text" is a serialization
    of the same tabular variables. To avoid a from-scratch 6-layer transformer
    overfitting ~110 training patients, the default depth is reduced to 2
    layers. Keep `use_text` honest in ablations: text and tabular share
    information here, so 'no_text' / 'no_tabular' ablation rows are reported
    with that caveat (see results_multimodel/notes.txt).
    """
    SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}

    def __init__(self, embed_dim: int = 512, max_len: int = 128,
                 num_layers: int = 2, num_heads: int = 8,
                 ff_dim: int = 1024, dropout: float = 0.1,
                 vocab_size: int = 4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.vocab: Dict[str, int] = dict(self.SPECIAL)
        self.vocab_size = vocab_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.tok_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

    def fit_vocab(self, texts: List[str]):
        from collections import Counter
        counter: Counter = Counter()
        for t in texts:
            for tok in re.findall(r"[a-zA-Z0-9]+", t.lower()):
                counter[tok] += 1
        n_special = len(self.SPECIAL)
        max_new = self.vocab_size - n_special
        for word, _ in counter.most_common(max_new):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        logging.info("OfflineTextEncoder vocab built: %d tokens "
                     "(from %d unique words)", len(self.vocab),
                     len(counter))

    def _tokenise(self, text: str) -> List[int]:
        tokens = [self.SPECIAL["[CLS]"]]
        for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            tokens.append(self.vocab.get(tok, self.SPECIAL["[UNK]"]))
            if len(tokens) >= self.max_len - 1:
                break
        tokens.append(self.SPECIAL["[SEP]"])
        pad_len = self.max_len - len(tokens)
        tokens += [self.SPECIAL["[PAD]"]] * pad_len
        return tokens[:self.max_len]

    def encode_texts(self, texts: List[str],
                     device: torch.device) -> torch.Tensor:
        ids = torch.tensor([self._tokenise(t) for t in texts],
                           dtype=torch.long, device=device)
        pad_mask = (ids == self.SPECIAL["[PAD]"])
        pos = torch.arange(self.max_len, device=device).unsqueeze(0)
        x = self.tok_embed(ids) + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        cls = self.pool(x[:, 0, :])
        out = self.proj(cls)
        return F.normalize(out, dim=-1)

    def forward(self, texts: List[str],
                device: torch.device) -> torch.Tensor:
        return self.encode_texts(texts, device)


# ═══════════════════════════════════════════════════════════════════════
# X-ray augmentation
# ═══════════════════════════════════════════════════════════════════════
class _XRayAugWrap:
    def __init__(self, base_tf):
        self.base_tf = base_tf
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=0),
            T.RandomAffine(degrees=0, translate=(0.04, 0.04), fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.base_tf(self.aug(img))


# ═══════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            pos_ratio = targets.mean().item() if targets.numel() else 0.5
            alpha = 1.0 - pos_ratio
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__(); self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    L_cls = a_f*focal + a_d*dice + a_b*BCE.

    FIX (class balancing): pos_weight is now OPTIONAL and only supplied when
    the chosen balance strategy includes 'loss'. Previously a WeightedRandomSampler
    AND pos_weight were both active, double-correcting the imbalance and
    destabilizing calibration on the 172-patient cohort.
    """
    def __init__(self, focal_weight: float = 0.60,
                 dice_weight: float = 0.05,
                 bce_weight: float = 0.35,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.focal = AdaptiveFocalLoss()
        self.dice = DiceLoss()
        if pos_weight is not None:
            pos_weight = float(np.clip(pos_weight, 0.1, 10.0))
        self.bce = (nn.BCEWithLogitsLoss() if pos_weight is None else
                    nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor([pos_weight],
                                                dtype=torch.float32)))
        self.w = {"focal": focal_weight,
                  "dice": dice_weight,
                  "bce": bce_weight}

    def to(self, device):
        super().to(device)
        if (isinstance(self.bce, nn.BCEWithLogitsLoss) and
                getattr(self.bce, "pos_weight", None) is not None):
            self.bce.pos_weight = self.bce.pos_weight.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return (self.w["focal"] * self.focal(logits, targets)
                + self.w["dice"] * self.dice(logits, targets)
                + self.w["bce"] * self.bce(logits, targets))


def smooth_labels(y: torch.Tensor, eps: float = 0.07) -> torch.Tensor:
    return y * (1.0 - eps) + 0.5 * eps


# ═══════════════════════════════════════════════════════════════════════
# BiomedCLIP loader
# ═══════════════════════════════════════════════════════════════════════
def _resolve_local_biomedclip_dir(model_name: str,
                                  cache_dir: Optional[str] = None
                                  ) -> Optional[str]:
    candidates = []
    if model_name:
        candidates.append(model_name)
    if cache_dir:
        candidates.append(cache_dir)
    candidates += [
        "./BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ]
    for cand in candidates:
        if not cand:
            continue
        p = (cand.replace("hf-hub:", "")
             if cand.startswith("hf-hub:./") else cand)
        if (os.path.isdir(p) and
                os.path.isfile(os.path.join(p, "open_clip_config.json"))):
            return os.path.abspath(p)
    return None


def _validate_biomedclip_local_dir(local_dir: str) -> Dict[str, Any]:
    required = ["open_clip_config.json", "open_clip_pytorch_model.bin"]
    optional = ["config.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.txt", "tokenizer.json"]
    missing = [f for f in required
               if not os.path.isfile(os.path.join(local_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"BiomedCLIP folder incomplete: {missing}\n"
            f"Required: {required}\nOptional: {optional}\n"
            f"Folder: {local_dir}")

    miss_opt = [f for f in optional
                if not os.path.isfile(os.path.join(local_dir, f))]
    if miss_opt:
        logging.warning(
            "BiomedCLIP optional text/tokenizer files missing: %s", miss_opt)

    return {
        "config": os.path.join(local_dir, "open_clip_config.json"),
        "weights": os.path.join(local_dir, "open_clip_pytorch_model.bin"),
        "hf_config": os.path.join(local_dir, "config.json"),
        "has_text_config": os.path.isfile(
            os.path.join(local_dir, "config.json")),
        "has_vocab": os.path.isfile(
            os.path.join(local_dir, "vocab.txt")),
        "has_tokenizer": (
            os.path.isfile(os.path.join(local_dir, "tokenizer.json")) or
            os.path.isfile(os.path.join(local_dir, "vocab.txt"))),
    }


def _count_vocab_lines(vocab_path: str, default: int = 30522) -> int:
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return int(default)


def _infer_biomedclip_text_vocab_size(weights_path: Optional[str], default: int = 30522) -> int:
    """Infer the text embedding vocabulary size required by the BiomedCLIP checkpoint."""
    if not weights_path or not os.path.isfile(weights_path):
        return int(default)
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            for raw_k, v in ckpt.items():
                k = str(raw_k)
                for pfx in ("module.", "model."):
                    if k.startswith(pfx):
                        k = k[len(pfx):]
                if k.endswith("text.transformer.embeddings.word_embeddings.weight") or \
                   k == "text.transformer.embeddings.word_embeddings.weight":
                    return int(v.shape[0])
                if k.endswith("transformer.embeddings.word_embeddings.weight") and \
                   "text" in k:
                    return int(v.shape[0])
    except Exception as e:
        logging.warning("Could not infer BiomedCLIP text vocab_size from checkpoint: %s", e)
    return int(default)


def _ensure_pubmedbert_config(local_dir: str, weights_path: Optional[str] = None) -> bool:
    """
    The manually downloaded BiomedCLIP folder often contains the OpenCLIP
    config, checkpoint, and tokenizer files, but not the HuggingFace
    PubMedBERT config.json. OpenCLIP needs that config to instantiate the
    text tower offline before the combined BiomedCLIP checkpoint is loaded.

    This helper writes the standard BERT-base PubMedBERT architecture config
    when tokenizer/vocab files are present. We still load all actual weights
    from open_clip_pytorch_model.bin; this config only defines the module
    shapes for offline construction.
    """
    cfg_path = os.path.join(local_dir, "config.json")
    vocab_path = os.path.join(local_dir, "vocab.txt")
    if not os.path.isfile(vocab_path):
        return False

    # Critical: the BiomedCLIP OpenCLIP checkpoint expects the PubMedBERT
    # embedding matrix size from open_clip_pytorch_model.bin. Some manually
    # downloaded folders have a vocab.txt with fewer lines (e.g. 28895), while
    # the checkpoint text embedding is 30522 x 768. If config.json uses the
    # vocab.txt line count, OpenCLIP builds the wrong text tower and full
    # checkpoint loading fails with a word_embeddings size mismatch.
    vocab_size = _infer_biomedclip_text_vocab_size(weights_path, default=30522)

    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            old_vs = int(existing.get("vocab_size", -1))
            if old_vs == int(vocab_size):
                return True
            existing["vocab_size"] = int(vocab_size)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            logging.warning(
                "Updated PubMedBERT config.json vocab_size from %s to %d to match BiomedCLIP checkpoint.",
                old_vs, vocab_size)
            return True
        except Exception as e:
            logging.warning("Could not update existing PubMedBERT config.json; recreating it: %s", e)

    cfg = {
        "architectures": ["BertModel"],
        "attention_probs_dropout_prob": 0.1,
        "classifier_dropout": None,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "torch_dtype": "float32",
        "transformers_version": "4.x",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": int(vocab_size),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logging.warning(
        "Created missing PubMedBERT config.json at %s (vocab_size=%d). "
        "The actual text weights are still loaded from open_clip_pytorch_model.bin.",
        cfg_path, vocab_size)
    return True


def _register_local_openclip_config(oc_module, local_dir: str,
                                    cfg_path: str,
                                    image_only: bool = True) -> str:
    with open(cfg_path, "r", encoding="utf-8") as f:
        hf_cfg = json.load(f)
    if "model_cfg" not in hf_cfg:
        raise ValueError(
            f"Invalid open_clip_config.json; missing 'model_cfg': {cfg_path}")

    # Deep-copy through JSON so we never mutate the downloaded config object.
    model_cfg = json.loads(json.dumps(hf_cfg["model_cfg"]))
    text_cfg = model_cfg.setdefault("text_cfg", {})

    if image_only:
        # Fallback path: build a tiny OpenCLIP text tower only so the model
        # object can exist; we then load visual.* weights only and use the
        # OfflineTextEncoder for clinical text.
        model_cfg["custom_text"] = False
        model_cfg["text_cfg"] = {
            "context_length": int(text_cfg.get("context_length", 77)),
            "vocab_size": int(text_cfg.get("vocab_size", 30522)),
            "width": int(text_cfg.get("width", 512)),
            "heads": int(text_cfg.get("heads", 8)),
            "layers": 1,
        }
        name = "local_biomedclip_image_only_vitb16"
    else:
        # Full BiomedCLIP path: keep proj='mlp' and pooler_type from the
        # official open_clip_config.json because removing them changes the
        # text projection shape and prevents complete checkpoint loading.
        text_cfg["hf_model_name"] = local_dir
        text_cfg["hf_tokenizer_name"] = local_dir
        model_cfg["text_cfg"] = text_cfg
        model_cfg["custom_text"] = True
        name = "local_biomedclip_pubmedbert_vitb16_full"

    cfg_root = os.path.join(local_dir, "_openclip_local_cfg")
    os.makedirs(cfg_root, exist_ok=True)
    with open(os.path.join(cfg_root, name + ".json"),
              "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2)
    try:
        from open_clip import factory as ocf
        if hasattr(ocf, "add_model_config"):
            ocf.add_model_config(cfg_root)
        elif hasattr(oc_module, "add_model_config"):
            oc_module.add_model_config(cfg_root)
    except Exception as e:
        logging.warning("Could not register local OpenCLIP config: %s", e)
    return name


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for k, v in state.items():
        kk = str(k)
        for pfx in ("module.", "model."):
            kk = kk[len(pfx):] if kk.startswith(pfx) else kk
        clean[kk] = v
    return clean


def _load_checkpoint_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unsupported BiomedCLIP checkpoint format: {weights_path}")
    return _clean_state_dict_keys(ckpt)


def _prefix_count(keys: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k in keys:
        p = k.split(".", 1)[0]
        out[p] = out.get(p, 0) + 1
    return out


def _load_complete_biomedclip_checkpoint(model: nn.Module,
                                         weights_path: str,
                                         require_text: bool = True,
                                         min_loaded_ratio: float = 0.80
                                         ) -> Dict[str, Any]:
    """Load all compatible tensors from the official BiomedCLIP checkpoint."""
    clean = _load_checkpoint_dict(weights_path)
    current = model.state_dict()
    compatible, skipped_shape, missing_in_model = {}, [], []

    for k, v in clean.items():
        if k in current and tuple(current[k].shape) == tuple(v.shape):
            compatible[k] = v
        elif k in current:
            skipped_shape.append((k, tuple(v.shape), tuple(current[k].shape)))
        else:
            missing_in_model.append(k)

    if not compatible:
        raise RuntimeError("No compatible tensors found in BiomedCLIP checkpoint.")

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    loaded_keys = list(compatible.keys())
    loaded_ratio = len(compatible) / max(1, len(current))
    has_visual = any(k.startswith("visual.") for k in loaded_keys)
    has_text = any(k.startswith("text.") for k in loaded_keys)

    info = {
        "loaded": len(compatible),
        "model_tensors": len(current),
        "checkpoint_tensors": len(clean),
        "loaded_ratio": float(loaded_ratio),
        "has_visual_weights": bool(has_visual),
        "has_text_weights": bool(has_text),
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "skipped_shape": len(skipped_shape),
        "missing_in_model": len(missing_in_model),
        "loaded_by_prefix": _prefix_count(loaded_keys),
    }

    logging.info(
        "Loaded COMPLETE BiomedCLIP checkpoint: %d/%d model tensors "
        "(ratio=%.3f) | text=%s visual=%s | skipped_shape=%d",
        info["loaded"], info["model_tensors"], info["loaded_ratio"],
        info["has_text_weights"], info["has_visual_weights"],
        info["skipped_shape"])
    logging.info("BiomedCLIP loaded tensor prefixes: %s", info["loaded_by_prefix"])

    if skipped_shape:
        logging.warning("BiomedCLIP skipped shape mismatches preview: %s", skipped_shape[:8])

    if not has_visual:
        raise RuntimeError("Complete BiomedCLIP load failed: no visual.* tensors loaded.")
    if require_text and not has_text:
        raise RuntimeError("Complete BiomedCLIP load failed: no text.* tensors loaded.")
    if loaded_ratio < float(min_loaded_ratio):
        raise RuntimeError(
            f"Complete BiomedCLIP load coverage too low: {loaded_ratio:.3f} < {min_loaded_ratio:.3f}")
    return info


def _load_visual_weights_only(model: nn.Module,
                               weights_path: str) -> Dict[str, Any]:
    clean = _load_checkpoint_dict(weights_path)
    current = model.state_dict()
    visual, skipped = {}, []
    for k, v in clean.items():
        if not (k.startswith("visual.") or k == "logit_scale"):
            continue
        if k in current and tuple(current[k].shape) == tuple(v.shape):
            visual[k] = v
        elif k in current:
            skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
    if not visual:
        raise RuntimeError(
            "No compatible visual.* weights in BiomedCLIP checkpoint.")
    missing, unexpected = model.load_state_dict(visual, strict=False)
    info = {
        "loaded": len(visual),
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "skipped_shape": len(skipped),
        "has_visual_weights": True,
        "has_text_weights": False,
    }
    logging.info(
        "Loaded BiomedCLIP image tower only: %d tensors | missing=%d unexpected=%d",
        len(visual), len(missing), len(unexpected))
    if skipped:
        logging.warning("Skipped BiomedCLIP visual shape mismatches: %s", skipped[:5])
    return info


def _create_openclip_model_and_transforms(open_clip, local_name: str,
                                          preprocess_cfg: Dict[str, Any],
                                          pretrained_path: Optional[str] = None,
                                          disable_hf_text_preload: bool = True):
    """
    Build OpenCLIP using the official BiomedCLIP local-loading pattern.

    Important fix:
    - For FULL BiomedCLIP, pass pretrained=<open_clip_pytorch_model.bin> during
      create_model_and_transforms. This makes OpenCLIP treat the file as the
      full CLIP checkpoint and prevents it from trying to load a separate
      HuggingFace pytorch_model.bin from the BiomedCLIP directory.
    - Also pass pretrained_text=False / pretrained_hf=False when supported, so
      the PubMedBERT tower is first created from config and then overwritten by
      the complete OpenCLIP checkpoint.
    """
    kwargs = {"pretrained": pretrained_path}
    if preprocess_cfg:
        kwargs["force_preprocess_cfg"] = preprocess_cfg
    if disable_hf_text_preload:
        # New OpenCLIP uses pretrained_text; older releases used pretrained_hf.
        kwargs["pretrained_text"] = False
        kwargs["pretrained_hf"] = False

    attempts = []
    attempts.append(dict(kwargs))

    # If this OpenCLIP version does not support one of these kwargs, remove it.
    k2 = dict(kwargs); k2.pop("pretrained_hf", None)
    attempts.append(k2)
    k3 = dict(kwargs); k3.pop("pretrained_text", None)
    attempts.append(k3)
    k4 = dict(kwargs); k4.pop("pretrained_text", None); k4.pop("pretrained_hf", None)
    attempts.append(k4)
    k5 = dict(k4); k5.pop("force_preprocess_cfg", None)
    attempts.append(k5)

    last_err = None
    seen = set()
    for kw in attempts:
        sig = tuple(sorted(kw.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        try:
            return open_clip.create_model_and_transforms(local_name, **kw)
        except TypeError as e:
            last_err = e
            continue
    raise last_err


def load_pretrained_biomedclip(model_name: str,
                               cache_dir: Optional[str] = None):
    """
    Offline BiomedCLIP loader.

    First tries to build the official full BiomedCLIP model and load the
    complete open_clip_pytorch_model.bin checkpoint (visual + PubMedBERT text
    + projection/logit scale). If the local folder is missing enough text
    resources to instantiate PubMedBERT, it falls back to the previous
    image-tower-only path and logs the fallback explicitly.
    """
    try:
        import open_clip
    except Exception as e:
        raise RuntimeError(
            f"open_clip_torch required. "
            f"Install: pip install open_clip_torch timm transformers\n{e}")

    local_dir = _resolve_local_biomedclip_dir(model_name, cache_dir)
    if local_dir is None:
        raise FileNotFoundError("BiomedCLIP local dir not found.")

    paths = _validate_biomedclip_local_dir(local_dir)
    with open(paths["config"], "r", encoding="utf-8") as f:
        preprocess_cfg = json.load(f).get("preprocess_cfg", {})

    logging.info("Loading local BiomedCLIP from %s", local_dir)

    # Preferred path: full BiomedCLIP checkpoint.
    if paths["has_vocab"] or paths["has_tokenizer"]:
        try:
            _ensure_pubmedbert_config(local_dir, paths["weights"])
            local_name = _register_local_openclip_config(
                open_clip, local_dir, paths["config"], image_only=False)
            model, train_tf, val_tf = _create_openclip_model_and_transforms(
                open_clip, local_name, preprocess_cfg,
                pretrained_path=paths["weights"],
                disable_hf_text_preload=True)
            # The call above follows the official local-loading route and loads
            # open_clip_pytorch_model.bin as the full CLIP checkpoint. Load once
            # more with coverage checks so the log explicitly confirms that both
            # visual.* and text.* tensors are present.
            load_info = _load_complete_biomedclip_checkpoint(
                model, paths["weights"], require_text=True, min_loaded_ratio=0.80)
            clip_tokenizer = open_clip.get_tokenizer(local_name)
            offline_text_encoder = None
            logging.info("BiomedCLIP loaded in FULL offline mode; no HF download used.")
            model._biomedclip_load_info = load_info
            return model, train_tf, val_tf, clip_tokenizer, offline_text_encoder, True
        except Exception as e:
            logging.warning(
                "Full offline BiomedCLIP load failed: %s. Falling back to "
                "image-tower-only BiomedCLIP + OfflineTextEncoder.", e)

    # Fallback path: visual tower only + local clinical text encoder.
    local_name = _register_local_openclip_config(
        open_clip, local_dir, paths["config"], image_only=True)
    model, train_tf, val_tf = _create_openclip_model_and_transforms(
        open_clip, local_name, preprocess_cfg,
        pretrained_path=None,
        disable_hf_text_preload=True)
    load_info = _load_visual_weights_only(model, paths["weights"])
    clip_tokenizer = None
    offline_text_encoder = OfflineTextEncoder(embed_dim=512)
    model._biomedclip_load_info = load_info
    logging.warning(
        "BiomedCLIP loaded in IMAGE-ONLY fallback mode; text uses OfflineTextEncoder. "
        "For complete BiomedCLIP, keep config.json/vocab/tokenizer files in %s.",
        local_dir)
    logging.info("BiomedCLIP loaded; no HF download used.")
    return model, train_tf, val_tf, clip_tokenizer, offline_text_encoder, False


# ═══════════════════════════════════════════════════════════════════════
# Audio encoder — ESResNeXt
# ═══════════════════════════════════════════════════════════════════════
class _LogMelSpectrogram(nn.Module):
    def __init__(self, sr=44100, n_fft=2048, hop_length=512,
                 n_mels=128, fmin=0.0, fmax=None,
                 power=2.0, eps=1e-6):
        super().__init__()
        self.sr = sr; self.n_fft = n_fft
        self.hop_length = hop_length; self.n_mels = n_mels
        self.power = power; self.eps = eps
        if fmax is None:
            fmax = sr / 2.0
        self.register_buffer(
            "mel_fb",
            self._build_mel_fb(sr, n_fft, n_mels, fmin, fmax),
            persistent=False)
        self.register_buffer(
            "window", torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _build_mel_fb(sr, n_fft, n_mels, fmin, fmax):
        def h2m(f): return 2595.0 * np.log10(1 + f / 700)
        def m2h(m): return 700 * (10 ** (m / 2595) - 1)
        mel_pts = np.linspace(h2m(fmin), h2m(fmax), n_mels + 2)
        bin_pts = np.clip(
            np.floor(
                (n_fft + 1) * m2h(mel_pts) / sr).astype(np.int64),
            0, n_fft // 2)
        n_bins = n_fft // 2 + 1
        fb = np.zeros((n_mels, n_bins), dtype=np.float32)
        for m in range(1, n_mels + 1):
            fl = int(bin_pts[m - 1]); fc = int(bin_pts[m])
            fr = int(bin_pts[m + 1])
            for k in range(fl, fc):
                fb[m - 1, k] = (k - fl) / max(1, fc - fl)
            for k in range(fc, fr):
                fb[m - 1, k] = (fr - k) / max(1, fr - fc)
        return torch.from_numpy(fb)

    def forward(self, waves):
        if waves.dim() == 3:
            waves = waves.squeeze(1)
        if waves.dim() == 1:
            waves = waves.unsqueeze(0)
        with safe_no_amp_context(waves.device):
            w32 = waves.float()
            window = self.window.to(device=w32.device, dtype=w32.dtype)
            mel_fb = self.mel_fb.to(device=w32.device, dtype=w32.dtype)
            spec = torch.stft(
                w32, n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window, center=True,
                return_complex=True)
            mel = torch.matmul(mel_fb,
                               spec.abs() ** self.power)
            lm = torch.log(mel + self.eps)
            mu = lm.mean(dim=(-2, -1), keepdim=True)
            sd = lm.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            lm = (lm - mu) / sd
        return lm.unsqueeze(1)


class _ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,
                 cardinality=32, base_width=4):
        super().__init__()
        d = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.conv1 = nn.Conv2d(in_planes, d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d, 3, stride=stride, padding=1,
                                groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, planes * self.expansion,
                                1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))
            if stride != 1 or in_planes != planes * self.expansion
            else nn.Identity())

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))


class ESResNeXt(nn.Module):
    def __init__(self, sr=44100, n_fft=2048, hop_length=512,
                 n_mels=128, cardinality=32, base_width=4,
                 layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
                 out_dim=512, stem_channels=32):
        super().__init__()
        self.front_end = _LogMelSpectrogram(
            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_planes = stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(stem_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.pool = nn.Identity()
        self.head = nn.Linear(
            512 * _ResNeXtBottleneck.expansion, out_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def _make_layer(self, planes, blocks, stride):
        layers = [_ResNeXtBottleneck(
            self.in_planes, planes, stride=stride,
            cardinality=self.cardinality,
            base_width=self.base_width)]
        self.in_planes = planes * _ResNeXtBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_ResNeXtBottleneck(
                self.in_planes, planes, stride=1,
                cardinality=self.cardinality,
                base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, waves):
        x = self.front_end(waves); x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = x.mean(dim=(2, 3))
        return torch.nan_to_num(self.head(x), nan=0.0, posinf=1e4, neginf=-1e4)


class PretrainedESResNeXtFBSPAudioEncoder(nn.Module):
    """
    ESResNeXt-fbsp audio encoder.

    W3 NOTE: reviewers ask for intermediate fine-tuning on ICBHI 2017 before
    using this on lung sounds. A helper `intermediate_finetune_icbhi.py` hook
    is documented in results_multimodel/notes.txt. If an ICBHI-adapted
    checkpoint exists, point `weights_path` at it; the loader transfers any
    compatible tensors automatically.
    """
    def __init__(self, repo_dir="./ESResNeXt-fbsp",
                 weights_path="./ESResNeXt-fbsp/ESResNeXtFBSP_AudioSet.pt",
                 out_dim=512, freeze_backbone=True,
                 audio_sr=44100, audio_n_fft=2048, audio_hop=512,
                 audio_n_mels=128, audio_cardinality=16,
                 audio_base_width=4,
                 audio_layers=(2, 2, 2, 2)):
        super().__init__()
        self.repo_dir = os.path.abspath(repo_dir)
        self.weights_path = os.path.abspath(weights_path)
        self.out_dim = int(out_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.audio_n_fft = int(audio_n_fft)
        self.audio_n_mels = int(audio_n_mels)

        if not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"ESResNeXt-fbsp repo not found: {self.repo_dir}")
        model_py = os.path.join(self.repo_dir, "model",
                                "esresnet_fbsp.py")
        if not os.path.isfile(model_py):
            raise FileNotFoundError(
                f"model/esresnet_fbsp.py not found under {self.repo_dir}")
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"ESResNeXt-fbsp weights not found: {self.weights_path}")
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        try:
            from model.esresnet_fbsp import ESResNeXtFBSP
        except Exception as e:
            raise RuntimeError(
                f"Cannot import ESResNeXtFBSP from {self.repo_dir}: {e}"
            ) from e

        self.backbone = self._build_backbone(
            ESResNeXtFBSP, audio_n_fft, audio_n_mels, audio_sr,
            audio_hop, audio_cardinality, audio_base_width, audio_layers)

        ckpt = torch.load(self.weights_path, map_location="cpu",
                          weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "model_state_dict",
                        "net", "network"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]; break
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported checkpoint format")

        cleaned = {}
        for k, v in state.items():
            nk = str(k)
            for pfx in ("module.", "model.", "backbone."):
                nk = nk[len(pfx):] if nk.startswith(pfx) else nk
            cleaned[nk] = v

        current = self.backbone.state_dict()
        compatible, skipped_shape = {}, []
        for k, v in cleaned.items():
            if k not in current:
                continue
            if tuple(current[k].shape) != tuple(v.shape):
                skipped_shape.append(
                    (k, tuple(v.shape), tuple(current[k].shape)))
                continue
            compatible[k] = v

        if not compatible:
            raise RuntimeError("No compatible tensors.")
        missing, unexpected = self.backbone.load_state_dict(
            compatible, strict=False)
        fbsp_loaded = [k for k in compatible if "fbsp" in k.lower()]
        fbsp_skipped = [k for k, _, _ in skipped_shape
                        if "fbsp" in k.lower()]
        logging.info(
            "Loaded ESResNeXt-fbsp weights | loaded=%d "
            "skipped_shape=%d | FBSP: loaded=%d skipped=%d",
            len(compatible), len(skipped_shape),
            len(fbsp_loaded), len(fbsp_skipped))

        if skipped_shape:
            expected_window = self.audio_n_fft // 2 + 1
            harmful = []
            for name, cs, ms in skipped_shape:
                if name == "window" and cs[0] == expected_window:
                    continue
                harmful.append((name, cs, ms))
            if harmful:
                preview = "; ".join(
                    f"{n}:ckpt{cs}->model{ms}"
                    for n, cs, ms in harmful[:8])
                logging.warning(
                    "Skipped shape-mismatched tensors: %s", preview)

        try:
            self._replace_head(out_dim=self.out_dim)
        except Exception as e:
            logging.warning(
                "Head replacement failed: %s. Using default head.", e)
            if (hasattr(self.backbone, "fc") and
                    isinstance(self.backbone.fc, nn.Linear)):
                self.backbone.fc = nn.Linear(
                    self.backbone.fc.in_features, out_dim)

        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = any(
                    h in name.lower()
                    for h in ("fc", "classifier", "head"))
            logging.info(
                "ESResNeXt-fbsp backbone frozen; only head trainable.")

    def _build_backbone(self, cls, n_fft, n_mels, sr, hop_length,
                        cardinality, base_width, layers):
        try:
            sig = inspect.signature(cls.__init__)
            allowed = set(sig.parameters.keys())
        except Exception:
            allowed = set()
        candidate_kw = {"num_classes": 309, "classes_num": 309,
                        "apply_attention": True, "pretrained": False}
        kwargs = {
            k: v for k, v in {
                "n_fft": n_fft, "n_mels": n_mels, "sr": sr,
                "hop_length": hop_length, "cardinality": cardinality,
                "base_width": base_width, "layers": layers,
                **candidate_kw}.items()
            if k in allowed}
        attempts = [kwargs,
                    {"num_classes": 309, "apply_attention": True},
                    {"num_classes": 309}, {}]
        last_err = None
        for kw in attempts:
            try:
                return cls(**kw)
            except TypeError as e:
                last_err = e
        raise RuntimeError(
            f"Could not instantiate ESResNeXtFBSP. "
            f"Last error: {last_err}")

    def _replace_head(self, out_dim: int):
        for attr in ("fc", "classifier", "head", "output", "final"):
            layer = getattr(self.backbone, attr, None)
            if isinstance(layer, nn.Linear):
                setattr(self.backbone, attr,
                        nn.Linear(layer.in_features, out_dim))
                logging.info(
                    "Replaced ESResNeXt-fbsp %s: %d->%d",
                    attr, layer.in_features, out_dim)
                return

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        if waves.dim() == 3 and waves.shape[1] == 1:
            waves = waves.squeeze(1)
        if waves.dim() != 2:
            waves = waves.reshape(waves.shape[0], -1)
        out = self.backbone(waves)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            for k in ("embedding", "features", "logits", "out"):
                if k in out:
                    out = out[k]; break
        if not torch.is_tensor(out):
            raise RuntimeError("Unexpected audio encoder output type")
        if out.dim() > 2:
            out = out.flatten(1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# Attention and fusion modules
# ═══════════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(feature_dim, feature_dim * 3, bias=False)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        if x.dim() != 3:
            raise ValueError(
                f"Expected [B,K,D], got {tuple(x.shape)}")
        b, k, d = x.shape
        if k == 0:
            return torch.zeros(b, d, device=x.device, dtype=x.dtype)
        if mask is not None:
            mask_bool = (mask > 0.5
                         if mask.dtype != torch.bool else mask)
            mask_bool = mask_bool.to(device=x.device)
            keep = mask_bool.sum(dim=1) > 0
            out = torch.zeros(b, d, device=x.device, dtype=x.dtype)
            if keep.any():
                out[keep] = self._attend(x[keep], mask_bool[keep])
            return torch.nan_to_num(out, nan=0.0)
        return self._attend(x, None)

    def _attend(self, x, mask_bool):
        b, k, d = x.shape
        qkv = (self.qkv(x)
               .reshape(b, k, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k_, v = qkv.unbind(0)
        attn = (q @ k_.transpose(-2, -1)) * self.scale
        if mask_bool is not None:
            attn = attn.masked_fill(
                (~mask_bool).unsqueeze(1).unsqueeze(2),
                torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(b, k, d)
        out = self.proj(out)
        if mask_bool is not None:
            m = mask_bool.unsqueeze(-1).to(out.dtype)
            return torch.nan_to_num(
                (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0),
                nan=0.0)
        return torch.nan_to_num(out.mean(dim=1), nan=0.0)


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 p_drop: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=p_drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, query, key_value,
                return_weights: bool = False):
        attn_out, attn_w = self.mha(
            query, key_value, key_value,
            need_weights=return_weights,
            average_attn_weights=True)
        x = self.norm1(query + attn_out)
        out = self.norm2(x + self.ffn(x))
        if return_weights:
            return out, attn_w
        return out, None


class ModalityWeightedFusion(nn.Module):
    def __init__(self, embed_dim: int, num_modalities: int = 3):
        super().__init__()
        self.modality_weights = nn.Parameter(
            torch.ones(num_modalities) / num_modalities)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, modalities: List[Optional[torch.Tensor]],
                masks: List[Optional[torch.Tensor]] = None):
        weights = F.softmax(self.modality_weights, dim=0)
        B, embed_dim = None, None
        for mod in modalities:
            if mod is not None:
                B, embed_dim = mod.shape[0], mod.shape[1]; break
        if B is None:
            B = 1
        if embed_dim is None:
            embed_dim = self.norm.normalized_shape[0]
        device = next(self.parameters()).device
        dtype = next((m.dtype for m in modalities if m is not None),
                     torch.float32)
        fused = torch.zeros(B, embed_dim, device=device, dtype=dtype)
        total_w = torch.zeros(1, device=device, dtype=dtype)
        for i, mod in enumerate(modalities):
            if mod is None:
                continue
            fused = fused + weights[i] * mod
            total_w = total_w + weights[i]
        if total_w > 0:
            return self.norm(fused / total_w)
        return fused


class TabularEmbedding(nn.Module):
    def __init__(self, num_features: int, embed_dim: int,
                 hidden_dims: List[int] = [128, 256]):
        super().__init__()
        self.num_features = num_features
        self.feature_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                          nn.LayerNorm(32), nn.Dropout(0.1))
            for _ in range(num_features)])
        dims = [num_features * 32] + hidden_dims + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(),
                           nn.LayerNorm(dims[i + 1]),
                           nn.Dropout(0.2)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                f"Tabular input must be [B, num_features], "
                f"got {x.shape}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Dim mismatch: expected {self.num_features}, "
                f"got {x.shape[1]}")
        embedded = torch.cat(
            [layer(x[:, i:i + 1])
             for i, layer in enumerate(self.feature_embeds)], dim=1)
        return F.normalize(self.mlp(embedded), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════
def _bin_value(v, step: int):
    try:
        return int(round(float(v) / step) * step)
    except Exception:
        return "N/A"


def _dicom_to_rgb(ds) -> Image.Image:
    """
    FIX (DICOM windowing): chest X-rays are NOT in Hounsfield units, so the
    previous CT lung/soft-tissue/bone windows produced garbage 3-channel
    images. Use the DICOM VOI WindowCenter/WindowWidth tags when present;
    otherwise fall back to robust percentile min-max. The single windowed
    grayscale plane is replicated to 3 channels (BiomedCLIP expects RGB).
    """
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr * float(getattr(ds, "RescaleSlope", 1.0)) +
           float(getattr(ds, "RescaleIntercept", 0.0)))

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    def _first(x):
        try:
            return float(x[0]) if isinstance(x, (list, tuple)) or \
                hasattr(x, "__len__") and not isinstance(x, str) else float(x)
        except Exception:
            return None

    c = _first(wc); w = _first(ww)
    if c is not None and w is not None and w > 1:
        lo, hi = c - w / 2.0, c + w / 2.0
        g = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        g = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        g = 1.0 - g
    rgb = np.stack([g, g, g], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _load_image_any(path: Optional[str]) -> Image.Image:
    img, _ok = _load_image_any_with_status(path)
    return img


def _load_image_any_with_status(path: Optional[str]) -> Tuple[Image.Image, bool]:
    blank = Image.new("RGB", (224, 224), (0, 0, 0))
    if path is None or not os.path.exists(path):
        return blank, False
    try:
        if (os.getenv("COCROSS_SKIP_DICOM", "0") == "1" and
                path.lower().endswith(".dcm")):
            return blank, False
        if path.lower().endswith(".dcm"):
            if not _HAVE_PYDICOM:
                logging.warning("DICOM file skipped because pydicom is unavailable: %s", path)
                return blank, False
            return _dicom_to_rgb(pydicom.dcmread(path)), True
        return Image.open(path).convert("RGB"), True
    except Exception as e:
        logging.warning("Image load failed %s: %s", path, e)
        return blank, False


def _load_audio_wave(path: Optional[str], sr: int,
                     seconds: float, train: bool) -> torch.Tensor:
    t = max(1, int(round(float(sr) * float(seconds))))
    if path is None or not os.path.exists(path):
        return torch.zeros(t, dtype=torch.float32)
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa required for audio.")
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        logging.warning("Audio load failed %s: %s", path, e)
        return torch.zeros(t, dtype=torch.float32)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] < t:
        wav = np.pad(wav, (0, t - wav.shape[0]), mode="constant")
    elif wav.shape[0] > t:
        start = (np.random.randint(0, wav.shape[0] - t + 1)
                 if train else (wav.shape[0] - t) // 2)
        wav = wav[start:start + t]
    return torch.from_numpy(
        np.ascontiguousarray(wav[:t], dtype=np.float32))


CORE_TAB_COLS = [
    "Age", "Charlson Comorbidity index",
    "APACHE II", "SOFA of the day"]
OPTIONAL_TAB_CANDIDATES = [
    "Hb", "CRP", "PCT", "SpO2", "PaO2", "FiO2", "PaO2/FiO2"]


def determine_tab_cols(all_cols: List[str]) -> List[str]:
    missing = [c for c in CORE_TAB_COLS if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return CORE_TAB_COLS + [c for c in OPTIONAL_TAB_CANDIDATES
                             if c in all_cols]


OPT_TEXT_STEPS = {
    "Hb": 1, "CRP": 5, "PCT": 1, "SpO2": 1,
    "PaO2": 5, "FiO2": 5, "PaO2/FiO2": 10}


def build_patient_text(row, present_opt: List[str]) -> str:
    """
    Single canonical text serialization used BOTH in the dataset and in
    OfflineTextEncoder vocabulary fitting.

    FIX (vocab mismatch): previously the dataset emitted Age/CCI/APACHE/SOFA +
    optional fields, but the vocab was fit on only the 4 core fields, so every
    optional field name mapped to [UNK]. This single helper guarantees the
    fitted vocabulary matches the text seen at encode time.
    """
    parts = [
        f"Age:{_bin_value(row['Age'], 5)}",
        f"CCI:{_bin_value(row['Charlson Comorbidity index'], 1)}",
        f"APACHE:{_bin_value(row['APACHE II'], 1)}",
        f"SOFA:{_bin_value(row['SOFA of the day'], 1)}"]
    for ck in present_opt:
        parts.append(
            f"{ck}:{_bin_value(row.get(ck, 'N/A'), OPT_TEXT_STEPS.get(ck, 1))}")
    return ", ".join(parts)


class CoCrossDataset(Dataset):
    def __init__(self, base_path, metadata_df, transform=None,
                 fixed_images=4, borrow_images="none",
                 image_subdirs=None, patient_images=None,
                 use_precomputed=False, img_embed_dir=None,
                 embed_dim=None, use_audio=True,
                 audio_subdir="AUDIO", fixed_audio=1,
                 audio_sr=44100, audio_seconds=10.0,
                 is_train=False, patient_audio=None,
                 tab_cols: Optional[List[str]] = None):
        super().__init__()
        self.base_path = base_path
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.fixed_images = int(fixed_images)
        self.borrow_images = borrow_images
        self.use_precomputed = bool(use_precomputed)
        self.img_embed_dir = img_embed_dir
        self.embed_dim = embed_dim
        self.image_subdirs = image_subdirs or ["XR"]
        self.use_audio = bool(use_audio)
        self.audio_subdir = audio_subdir
        self.fixed_audio = int(fixed_audio)
        self.audio_sr = int(audio_sr)
        self.audio_seconds = float(audio_seconds)
        self.is_train = bool(is_train)
        self.similar_patients = None
        self.tab_cols = (tab_cols if tab_cols is not None
                         else determine_tab_cols(
                             list(metadata_df.columns)))
        self.expected_tab_dim = len(self.tab_cols)
        self.present_opt = [c for c in OPTIONAL_TAB_CANDIDATES
                            if c in self.df.columns]
        if self.borrow_images == "same_label":
            fs = StandardScaler().fit_transform(
                self.df[CORE_TAB_COLS].values)
            from sklearn.metrics.pairwise import euclidean_distances
            dm = euclidean_distances(fs)
            self.similar_patients = [
                np.argsort(dm[i])[1:] for i in range(len(self.df))]
        self.patient_images = (patient_images
                               if patient_images is not None
                               else self._scan_images())
        self.patient_audio = (patient_audio
                              if patient_audio is not None
                              else self._scan_audio())
        self.data = self._prepare_data()

    def _scan_images(self):
        out = {}
        for _, row in self.df.iterrows():
            pid = int(row["Patient number"]); paths = []
            for sub in self.image_subdirs or []:
                d = os.path.join(self.base_path, str(pid), sub)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(
                                (".png", ".jpg", ".jpeg", ".dcm")):
                            paths.append(os.path.join(d, f))
            out[pid] = sorted(paths)
        return out

    def _scan_audio(self):
        if not self.use_audio:
            return {}
        out = {}; exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
        for _, row in self.df.iterrows():
            pid = int(row["Patient number"])
            d = os.path.join(
                self.base_path, str(pid), self.audio_subdir)
            paths = ([os.path.join(d, f)
                      for f in os.listdir(d)
                      if f.lower().endswith(exts)]
                     if os.path.isdir(d) else [])
            out[pid] = sorted(paths)
        return out

    @staticmethod
    def _key_for_path(path, root=None):
        abspath = os.path.abspath(path)
        try:
            rel = (os.path.relpath(abspath, os.path.abspath(root))
                   if root else abspath)
        except Exception:
            rel = abspath
        return hashlib.sha1(rel.encode("utf-8")).hexdigest() + ".pt"

    @staticmethod
    def _choose_fixed(pool, k, seed):
        if len(pool) >= k:
            idx = np.random.default_rng(seed=seed).choice(
                len(pool), size=k, replace=False)
            return [pool[i] for i in idx]
        chosen = list(pool)
        while len(chosen) < k:
            chosen.append(None)
        return chosen

    def _find_similar_patients_with_modality(self, pid, has_img,
                                              has_aud, k=2):
        if not self.tab_cols:
            return []
        clinical = self.df[self.tab_cols].values.astype(np.float32)
        clinical = np.nan_to_num(
            clinical,
            nan=np.nanmean(clinical, axis=0))
        scaler = StandardScaler()
        clinical_scaled = scaler.fit_transform(clinical)
        target_mask = self.df["Patient number"] == pid
        if not target_mask.any():
            return []
        target_idx = np.where(target_mask)[0][0]
        distances = np.linalg.norm(
            clinical_scaled - clinical_scaled[target_idx], axis=1)
        target_label = self.df.iloc[target_idx]["90 days survival"]
        candidates = []
        for idx in np.argsort(distances)[1:]:
            if len(candidates) >= k:
                break
            cand_pid = int(self.df.iloc[idx]["Patient number"])
            if self.df.iloc[idx]["90 days survival"] != target_label:
                continue
            if has_img and not self.patient_images.get(cand_pid, []):
                continue
            if has_aud and not self.patient_audio.get(cand_pid, []):
                continue
            candidates.append(cand_pid)
        return candidates

    def _prepare_data(self):
        present_opt = self.present_opt
        data_list = []
        for pos, (_, row) in enumerate(self.df.iterrows()):
            pid = int(row["Patient number"])
            pool_img = list(self.patient_images.get(pid, []))
            pool_aud = list(self.patient_audio.get(pid, []))
            if (len(pool_img) < self.fixed_images and
                    self.borrow_images == "clinical_similar"):
                for sim_pid in self._find_similar_patients_with_modality(
                        pid, True, False, k=2):
                    sim_pool = self.patient_images.get(sim_pid, [])
                    if sim_pool and len(pool_img) < self.fixed_images:
                        pool_img.append(
                            np.random.default_rng(
                                seed=pid + sim_pid).choice(sim_pool))
            if (len(pool_aud) < self.fixed_audio and
                    self.use_audio and
                    self.borrow_images == "clinical_similar"):
                for sim_pid in self._find_similar_patients_with_modality(
                        pid, False, True, k=2):
                    sim_pool = self.patient_audio.get(sim_pid, [])
                    if sim_pool and len(pool_aud) < self.fixed_audio:
                        pool_aud.append(
                            np.random.default_rng(
                                seed=pid * 17 + sim_pid).choice(
                                sim_pool))
            if (len(pool_img) < self.fixed_images and
                    self.borrow_images == "same_label" and
                    self.similar_patients is not None):
                chosen = list(pool_img)
                missing = self.fixed_images - len(chosen)
                lbl = int(row["90 days survival"])
                for si in self.similar_patients[pos]:
                    sr2 = self.df.iloc[si]
                    if int(sr2["90 days survival"]) != lbl:
                        continue
                    sp = int(sr2["Patient number"])
                    pool2 = self.patient_images.get(sp, [])
                    if pool2 and missing > 0:
                        chosen.append(
                            np.random.default_rng(
                                seed=pid + sp).choice(pool2))
                        missing -= 1
                    if missing == 0:
                        break
                while len(chosen) < self.fixed_images:
                    chosen.append(None)
                chosen_img = chosen
            else:
                chosen_img = self._choose_fixed(
                    pool_img, self.fixed_images, seed=pid)
            chosen_aud = (
                self._choose_fixed(pool_aud, self.fixed_audio,
                                   seed=pid * 17 + 3)
                if self.use_audio else [])
            rec = {
                "patient_id": pid,
                "text": build_patient_text(row, present_opt),
                "label": float(row["90 days survival"]),
                "tab": (np.asarray(row["_tab"], dtype=np.float32)
                        if "_tab" in row else None)}
            if self.use_precomputed:
                rec["embed_keys"] = [
                    None if p is None else
                    self._key_for_path(p, self.base_path)
                    for p in chosen_img]
            else:
                rec["image_paths"] = chosen_img
            if self.use_audio:
                rec["audio_paths"] = chosen_aud
            data_list.append(rec)
        return data_list

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.use_precomputed:
            feats, mask_img = [], []
            for k in item["embed_keys"]:
                if k is None or self.img_embed_dir is None:
                    if self.embed_dim is None:
                        raise RuntimeError("embed_dim required")
                    feats.append(torch.zeros(self.embed_dim))
                    mask_img.append(0.0)
                else:
                    p = os.path.join(self.img_embed_dir, k)
                    if os.path.exists(p):
                        feats.append(
                            torch.load(p, map_location="cpu",
                                       weights_only=True).float()
                            .squeeze(0))
                        mask_img.append(1.0)
                    else:
                        feats.append(torch.zeros(self.embed_dim))
                        mask_img.append(0.0)
            visual_query = torch.stack(feats, dim=0)
        else:
            ims, mask_img = [], []
            for p in item.get("image_paths", []):
                im, ok = _load_image_any_with_status(p)
                ims.append(im)
                mask_img.append(1.0 if ok else 0.0)
            if self.transform:
                ims = [self.transform(im) for im in ims]
            visual_query = (torch.stack(ims) if ims
                            else torch.zeros(0))
        image_mask = torch.tensor(mask_img, dtype=torch.float32)
        audio_query = torch.zeros(0)
        audio_mask = torch.zeros(0)
        if self.use_audio:
            waves, mask_aud = [], []
            for p in item.get("audio_paths", []):
                w = _load_audio_wave(
                    p, sr=self.audio_sr,
                    seconds=self.audio_seconds, train=self.is_train)
                waves.append(w)
                mask_aud.append(
                    0.0 if p is None
                    else float(w.abs().sum().item() > 0))
            if waves:
                audio_query = torch.stack(waves, dim=0)
                audio_mask = torch.tensor(mask_aud,
                                          dtype=torch.float32)
        out = {
            "patient_id": int(item["patient_id"]),
            "visual_query": visual_query,
            "image_mask": image_mask,
            "audio_query": audio_query,
            "audio_mask": audio_mask,
            "textual_query": item["text"],
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "precomputed": torch.tensor(
                [1.0 if self.use_precomputed else 0.0])}
        if item.get("tab") is not None:
            tab_arr = np.asarray(item["tab"], dtype=np.float32).ravel()
            if len(tab_arr) != self.expected_tab_dim:
                logging.warning(
                    "Patient %d: tab shape mismatch. Zero tensor.",
                    item["patient_id"])
                out["tab_features"] = torch.zeros(
                    self.expected_tab_dim, dtype=torch.float32)
            else:
                out["tab_features"] = torch.from_numpy(tab_arr)
        else:
            out["tab_features"] = torch.zeros(
                self.expected_tab_dim, dtype=torch.float32)
        return out


# ═══════════════════════════════════════════════════════════════════════
# Sampling / metadata utilities
# ═══════════════════════════════════════════════════════════════════════
def get_sample_weights(labels: np.ndarray,
                       strategy: str = "effective") -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    if strategy == "balanced":
        w = 1.0 / counts
    elif strategy == "sqrt":
        w = 1.0 / np.sqrt(counts)
    elif strategy == "log":
        w = 1.0 / np.log(counts + 1.0)
    elif strategy == "effective":
        eff = (1.0 - np.power(0.9999, counts)) / (1.0 - 0.9999)
        w = 1.0 / eff
    else:
        w = np.ones_like(counts)
    return w[labels]


def _resolve_metadata_path(metadata_path: str) -> str:
    candidates = [metadata_path]
    root, _ = os.path.splitext(metadata_path)
    for e in [".xlsx", ".xls", ".csv"]:
        cand = root + e
        if cand not in candidates:
            candidates.append(cand)
    base = os.path.dirname(metadata_path) or "."
    for name in ["Metadata.xlsx", "Metadata.xls", "Metadata.csv",
                 "metadata.xlsx", "metadata.xls", "metadata.csv"]:
        cand = os.path.join(base, name)
        if cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("Metadata not found.")


def _read_metadata_table(metadata_path: str) -> pd.DataFrame:
    metadata_path = _resolve_metadata_path(metadata_path)
    ext = os.path.splitext(metadata_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(metadata_path)
        except ImportError as e:
            raise RuntimeError("openpyxl required") from e
    if ext == ".csv":
        return pd.read_csv(metadata_path, encoding="utf-8",
                           engine="python")
    raise ValueError(f"Unsupported metadata extension {ext!r}")


def _canonicalize_metadata_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c
            for c in data.columns}
    aliases = {
        "Patient number": [
            "patientnumber", "patientid", "patient", "pid",
            "id", "subjectid"],
        "90 days survival": [
            "90dayssurvival", "survival90days", "survival",
            "outcome", "label", "mortality90days"],
        "Age": ["age", "patientage"],
        "Charlson Comorbidity index": [
            "charlsoncomorbidityindex", "charlson", "cci"],
        "APACHE II": ["apacheii", "apache2"],
        "SOFA of the day": ["sofaoftheday", "sofa"],
        "PaO2/FiO2": ["pao2fio2", "pafi"],
        "Sex": ["sex", "gender"],
    }
    rename = {}
    for target, keys in aliases.items():
        if target in data.columns:
            continue
        for key in keys:
            k = re.sub(r"[^a-z0-9]+", "", key.lower())
            if k in norm:
                rename[norm[k]] = target; break
    if rename:
        data = data.rename(columns=rename)
    return data


def load_full_metadata(
        metadata_path: str) -> Tuple[pd.DataFrame, List[str]]:
    data = _canonicalize_metadata_columns(
        _read_metadata_table(metadata_path))
    tab_cols = determine_tab_cols(list(data.columns))
    missing_req = [c for c in ["Patient number", "90 days survival"]
                   if c not in data.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}.")
    keep_extra = [c for c in ["Sex"] if c in data.columns]
    df = data[["Patient number", "90 days survival"] +
              tab_cols + keep_extra].copy()
    for c in tab_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _map_label(x):
        s = re.sub(r"\s+", " ", str(x).strip().lower())
        try:
            v = float(s)
            if v in (0.0, 1.0):
                return int(v)
        except Exception:
            pass
        if any(k in s for k in ["died", "expired",
                                  "deceased", "nonsurvivor"]):
            return 0
        if any(k in s for k in ["lived", "survivor",
                                  "alive", "discharged"]):
            return 1
        tokens = set(re.findall(r"[a-z]+", s))
        if tokens & {"alive", "survived", "yes", "true", "positive"}:
            return 1
        if tokens & {"dead", "died", "no", "false",
                      "negative", "mortality"}:
            return 0
        return np.nan

    df["90 days survival"] = (
        df["90 days survival"].apply(_map_label).astype("float"))
    df = df.dropna(subset=["90 days survival"]).copy()
    df["90 days survival"] = df["90 days survival"].astype(int)
    df["Patient number"] = pd.to_numeric(
        df["Patient number"], errors="coerce")
    df = df.dropna(subset=["Patient number"]).copy()
    df["Patient number"] = df["Patient number"].astype(int)
    return df, tab_cols


def apply_tabular_transform(tab_cols, df_, imputer, scaler):
    df2 = df_.copy()
    for c in tab_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2["_tab"] = list(
        scaler.transform(
            imputer.transform(df2[tab_cols])).astype(np.float32))
    return df2


def attach_tabular(tab_cols, df_train, df_val):
    df_train = df_train.copy(); df_val = df_val.copy()
    for df_ in (df_train, df_val):
        for c in tab_cols:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler().fit(
        imputer.fit_transform(df_train[tab_cols]))
    return (apply_tabular_transform(tab_cols, df_train,
                                    imputer, scaler),
            apply_tabular_transform(tab_cols, df_val,
                                    imputer, scaler),
            imputer, scaler)


def scan_all_patient_images(base_path, df_all, image_subdirs):
    out = {}
    for pid in df_all["Patient number"].astype(int).tolist():
        paths = []
        for sub in image_subdirs or []:
            d = os.path.join(base_path, str(pid), sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".dcm")):
                        paths.append(os.path.join(d, f))
        out[int(pid)] = sorted(paths)
    return out


def _pid_tokens(pid: int) -> List[str]:
    p = int(pid)
    return [str(p), f"{p:03d}", f"{p:04d}",
            f"patient{p}", f"patient_{p}",
            f"pid{p}", f"pid_{p}"]


def _path_matches_pid(path: str, pid: int) -> bool:
    low = os.path.basename(path).lower()
    parent = os.path.basename(os.path.dirname(path)).lower()
    grand = os.path.basename(
        os.path.dirname(os.path.dirname(path))).lower()
    hay = " ".join([low, parent, grand])
    return any(
        re.search(
            rf"(?<![a-z\d]){re.escape(tok.lower())}(?![a-z\d])",
            hay)
        for tok in _pid_tokens(pid))


def scan_all_patient_audio(
        base_path: str, df_all: pd.DataFrame,
        audio_subdir: str = "AUDIO",
        audio_root: Optional[str] = None,
        extra_audio_roots: Optional[List[str]] = None
) -> Dict[int, List[str]]:
    exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    pids = [int(x) for x in df_all["Patient number"].astype(int)]
    out: Dict[int, List[str]] = {}

    def _walk(root):
        if not root or not os.path.isdir(root):
            return []
        res = []
        for dp, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(exts):
                    res.append(os.path.join(dp, f))
        return sorted(res)

    for pid in pids:
        found = []
        for d in [
            os.path.join(base_path, str(pid), audio_subdir),
            os.path.join(base_path, f"{pid:03d}", audio_subdir),
            os.path.join(base_path, f"{pid:04d}", audio_subdir),
            os.path.join(base_path, str(pid)),
        ]:
            found.extend(_walk(d))
        unique = []; seen = set()
        for f in sorted(found):
            af = os.path.abspath(f)
            if af not in seen:
                unique.append(f); seen.add(af)
        out[int(pid)] = unique

    if all(len(out[pid]) > 0 for pid in pids):
        return out
    global_roots = [audio_root] if audio_root else []
    if extra_audio_roots:
        global_roots.extend(extra_audio_roots)
    global_roots.extend([
        "audio", "./audio",
        os.path.join(base_path, "audio"),
        os.path.join(base_path, "AUDIO")])
    global_files = []; seen_roots = set()
    for root in global_roots:
        ar = os.path.abspath(root)
        if ar in seen_roots:
            continue
        seen_roots.add(ar); global_files.extend(_walk(root))
    if global_files:
        for pid in pids:
            if len(out[pid]) > 0:
                continue
            matches = [f for f in global_files
                       if _path_matches_pid(f, pid)]
            unique = list(out.get(pid, []))
            seen = {os.path.abspath(f) for f in unique}
            for f in sorted(matches):
                af = os.path.abspath(f)
                if af not in seen:
                    unique.append(f); seen.add(af)
            out[int(pid)] = unique
    return out


def log_modality_coverage(df_all, patient_images, patient_audio,
                           use_audio=True):
    pids = df_all["Patient number"].astype(int).tolist()
    n = max(1, len(pids))
    n_img = sum(1 for pid in pids
                if len(patient_images.get(int(pid), [])) > 0)
    n_aud = sum(1 for pid in pids
                if use_audio and
                len(patient_audio.get(int(pid), [])) > 0)
    logging.info(
        "Modality coverage | clinical=%d/%d | X-ray=%d/%d | "
        "audio=%d/%d", len(pids), n, n_img, n, n_aud, n)
    print(f"Modality coverage | clinical={len(pids)}/{n} | "
          f"X-ray={n_img}/{n} | audio={n_aud}/{n}")


def collapse_to_patient_level(df, tab_cols, agg="mean",
                               label_mode="majority",
                               time_col=None):
    df = df.copy()
    has_sex = "Sex" in df.columns
    if agg in {"mean", "median", "max", "min"}:
        tab_agg = df.groupby("Patient number")[tab_cols].agg(agg)
    elif agg in {"last", "first"}:
        if time_col is None:
            raise ValueError("agg='last'/'first' requires time_col.")
        df = df.sort_values(["Patient number", time_col])
        pick = (df.groupby("Patient number").tail(1)
                if agg == "last"
                else df.groupby("Patient number").head(1))
        tab_agg = pick.groupby("Patient number")[tab_cols].first()
    else:
        raise ValueError(f"Unknown agg='{agg}'")

    def _reduce(s):
        if label_mode == "majority":
            return int(round(s.mean()))
        if label_mode == "any_positive":
            return int(s.max())
        if label_mode == "all_positive":
            return int(s.min())
        if label_mode == "unique":
            if s.nunique() != 1:
                raise ValueError(
                    f"Inconsistent labels: {list(s.unique())}")
            return int(s.iloc[0])
        raise ValueError(f"Unknown label_mode='{label_mode}'")

    out = tab_agg.join(
        df.groupby("Patient number")["90 days survival"].agg(
            _reduce))
    if has_sex:
        sex_map = df.groupby("Patient number")["Sex"].agg(
            lambda s: s.iloc[0])
        out = out.join(sex_map)
    out = out.reset_index()
    out["Patient number"] = out["Patient number"].astype(int)
    out["90 days survival"] = out["90 days survival"].astype(int)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Main model — BiomedAudioCLIP
# ═══════════════════════════════════════════════════════════════════════
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = copy.deepcopy(model)
        if device is not None:
            self.ema.to(device=device)
        self.ema.eval(); self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        src = (model._orig_mod
               if hasattr(model, "_orig_mod") else model)
        ep = dict(self.ema.named_parameters())
        mp = dict(src.named_parameters())
        for k in ep:
            if k in mp:
                ep[k].data.mul_(self.decay).add_(
                    mp[k].data, alpha=1.0 - self.decay)
        eb = dict(self.ema.named_buffers())
        mb = dict(src.named_buffers())
        for k in eb:
            if k not in mb:
                continue
            if eb[k].dtype.is_floating_point:
                eb[k].data.mul_(self.decay).add_(
                    mb[k].data, alpha=1.0 - self.decay)
            else:
                eb[k].copy_(mb[k])


class BiomedAudioCLIP(nn.Module):
    """
    BiomedAudioCLIP — tri-modal ICU survival prediction model.

    Modalities:
      Image  : BiomedCLIP ViT-B/16 + Masked Set-Attention
      Text   : BiomedCLIP BERT OR OfflineTextEncoder
      Audio  : ESResNeXt-50 FBSP + Masked Set-Attention
      Tabular: per-feature MLP + mixer
    """
    def __init__(self, clip_model, clip_tokenizer,
                 offline_text_encoder=None,
                 embed_dim=512, hidden_dim=512,
                 dropout=0.30,
                 freeze_biomedclip=True,
                 use_text=True,
                 use_tab=True, use_audio=True,
                 tab_dim=7, num_heads_img=8,
                 num_heads_audio=4, num_heads_xattn=4,
                 p_drop_text=0.0, p_drop_tab=0.10,
                 p_drop_audio=0.10, use_cross_attn=True,
                 audio_sr=44100, audio_n_fft=2048,
                 audio_hop=512, audio_n_mels=128,
                 audio_cardinality=16, audio_base_width=4,
                 audio_layers=(2, 2, 2, 2),
                 use_pretrained_esresnext=True,
                 esresnext_repo_dir="./ESResNeXt-fbsp",
                 esresnext_weights_path=(
                     "./ESResNeXt-fbsp/ESResNeXtFBSP_AudioSet.pt"),
                 freeze_esresnext_backbone=True,
                 init_clip_temp=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.offline_text_encoder = offline_text_encoder
        self.embed_dim = int(embed_dim)
        self.freeze_biomedclip = bool(freeze_biomedclip)
        self.base_use_text = bool(use_text)
        self.base_use_tab = bool(use_tab)
        self.base_use_audio = bool(use_audio)
        self.p_drop_text = float(p_drop_text)
        self.p_drop_tab = float(p_drop_tab)
        self.p_drop_audio = float(p_drop_audio)
        self.use_cross_attn = (bool(use_cross_attn) and
                               (use_text or use_tab or use_audio))
        self._unfreeze_epoch = None

        if self.freeze_biomedclip:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

        self.image_attention = MultiHeadAttention(
            self.embed_dim, num_heads=num_heads_img, dropout=0.1)
        self.audio_encoder = None
        self.audio_attention = None
        if self.base_use_audio:
            if use_pretrained_esresnext:
                self.audio_encoder = \
                    PretrainedESResNeXtFBSPAudioEncoder(
                        repo_dir=esresnext_repo_dir,
                        weights_path=esresnext_weights_path,
                        out_dim=self.embed_dim,
                        freeze_backbone=freeze_esresnext_backbone,
                        audio_sr=audio_sr, audio_n_fft=audio_n_fft,
                        audio_hop=audio_hop,
                        audio_n_mels=audio_n_mels,
                        audio_cardinality=audio_cardinality,
                        audio_base_width=audio_base_width,
                        audio_layers=audio_layers)
            else:
                self.audio_encoder = ESResNeXt(
                    sr=audio_sr, n_fft=audio_n_fft,
                    hop_length=audio_hop, n_mels=audio_n_mels,
                    cardinality=audio_cardinality,
                    base_width=audio_base_width,
                    layers=tuple(audio_layers),
                    out_dim=self.embed_dim)
            self.audio_attention = MultiHeadAttention(
                self.embed_dim, num_heads=num_heads_audio,
                dropout=0.1)

        if self.base_use_tab:
            assert tab_dim > 0, \
                f"tab_dim must be > 0 when use_tab=True, got {tab_dim}"
            self.tab_embed = TabularEmbedding(
                tab_dim, self.embed_dim, [128, 256])

        if self.use_cross_attn:
            self.cross_attention = CrossModalAttention(
                self.embed_dim, num_heads=num_heads_xattn,
                p_drop=0.1)

        self.modality_fusion = ModalityWeightedFusion(
            self.embed_dim, num_modalities=3)

        self.fusion1 = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout))
        self.fusion2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2,
                      max(16, hidden_dim // 4)),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 4), 1))

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(
                1.0 / max(1e-6, init_clip_temp)))

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_biomedclip", False):
            self.clip_model.eval()
        # FIX (BatchNorm singleton-batch crash): a frozen pretrained audio
        # backbone must stay in eval mode so its BatchNorm layers use the
        # pretrained running statistics instead of recomputing batch stats.
        # In training mode, a 1-sample last batch produces a [1, C, 1, 1]
        # feature map and BN raises "Expected more than 1 value per channel".
        # eval-mode BN also accepts batch size 1. The trainable head (a Linear)
        # is unaffected by eval mode and still receives gradients.
        if (getattr(self, "audio_encoder", None) is not None and
                getattr(self.audio_encoder, "freeze_backbone", False) and
                hasattr(self.audio_encoder, "backbone")):
            self.audio_encoder.backbone.eval()
        return self

    def progressive_unfreeze(self, epoch: int,
                              unfreeze_at: int = 30,
                              enabled: bool = False):
        if not enabled:
            return 0
        if epoch != unfreeze_at or self._unfreeze_epoch == epoch:
            return 0
        self._unfreeze_epoch = epoch
        unfrozen = 0
        if hasattr(self.clip_model, "visual") and \
                hasattr(self.clip_model.visual, "transformer"):
            blocks = self.clip_model.visual.transformer.resblocks
            for i, block in enumerate(blocks):
                if i >= 11:
                    for p in block.parameters():
                        p.requires_grad = True
                        unfrozen += 1
        if self.audio_encoder is not None and \
                hasattr(self.audio_encoder, "backbone"):
            if hasattr(self.audio_encoder.backbone, "layer4"):
                for p in self.audio_encoder.backbone.layer4.parameters():
                    p.requires_grad = True
                    unfrozen += 1
        logging.info(
            "Progressive unfreeze at epoch %d: unfrozen %d parameter tensors "
            "(enabled=%s)", epoch, unfrozen, enabled)
        return unfrozen

    def _encode_image_raw(self, x):
        ctx = (torch.no_grad()
               if self.freeze_biomedclip else nullcontext())
        with ctx:
            feats = self.clip_model.encode_image(x)
        feats = torch.nan_to_num(feats.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.nan_to_num(F.normalize(feats, dim=-1), nan=0.0)

    def _encode_text_raw(self, texts: List[str],
                          device: torch.device) -> torch.Tensor:
        if self.offline_text_encoder is not None:
            return self.offline_text_encoder(texts, device)
        tokens = self.clip_tokenizer(list(texts)).to(device)
        ctx = (torch.no_grad()
               if self.freeze_biomedclip else nullcontext())
        with ctx:
            feats = self.clip_model.encode_text(tokens)
        feats = torch.nan_to_num(feats.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.nan_to_num(F.normalize(feats, dim=-1), nan=0.0)

    def extract_audio_fea(self, audio_query, audio_mask, device):
        if (self.audio_encoder is None or audio_query is None or
                audio_query.numel() == 0):
            return None
        b, k_a = audio_query.shape[0], audio_query.shape[1]
        if k_a == 0:
            return torch.zeros(b, self.embed_dim, device=device)
        waves = audio_query.reshape(b * k_a, -1).to(
            device, non_blocking=True)
        raw_audio = sanitize_tensor(self.audio_encoder(waves), clamp=1e4)
        a_emb = torch.nan_to_num(
            F.normalize(raw_audio, dim=-1), nan=0.0
        ).reshape(b, k_a, self.embed_dim)
        if audio_mask is None or audio_mask.numel() == 0:
            pooled = a_emb.mean(dim=1)
        else:
            pooled = self.audio_attention(
                a_emb, audio_mask.to(device, non_blocking=True))
        # FIX: renormalize after set-attention pooling so the audio embedding
        # is unit-norm like image/text embeddings. Previously attention output
        # was un-normalized, making the contrastive (CLIP) loss inconsistent
        # across modalities (cosine sim assumes unit vectors).
        return torch.nan_to_num(F.normalize(pooled, dim=-1), nan=0.0)

    def forward(self, textual_query, visual_query,
                image_mask=None, tab_features=None,
                audio_query=None, audio_mask=None,
                return_embeds=False,
                return_attn_weights=False):
        if (isinstance(visual_query, torch.Tensor) and
                visual_query.numel()):
            device = visual_query.device
        elif isinstance(tab_features, torch.Tensor):
            device = tab_features.device
        elif (isinstance(audio_query, torch.Tensor) and
              audio_query.numel()):
            device = audio_query.device
        else:
            device = next(self.parameters()).device

        if (visual_query is not None and visual_query.numel()):
            if (visual_query.dim() == 3 and
                    visual_query.shape[-1] == self.embed_dim):
                if image_mask is not None:
                    image_mask = image_mask.to(device,
                                               non_blocking=True)
                img_pooled = self.image_attention(
                    visual_query.to(device, non_blocking=True),
                    image_mask)
            else:
                b, k = visual_query.shape[:2]
                imgs = visual_query.reshape(
                    b * k, *visual_query.shape[2:]).to(
                    device, non_blocking=True)
                if image_mask is not None:
                    image_mask = image_mask.to(device,
                                               non_blocking=True)
                    valid = image_mask.reshape(b * k) > 0.5
                else:
                    valid = torch.ones(b * k, dtype=torch.bool,
                                       device=device)
                img_flat = torch.zeros(b * k, self.embed_dim,
                                       device=device)
                if valid.any():
                    img_flat[valid] = self._encode_image_raw(
                        imgs[valid])
                img_pooled = self.image_attention(
                    img_flat.reshape(b, k, self.embed_dim),
                    image_mask)
        else:
            b = (audio_query.shape[0]
                 if isinstance(audio_query, torch.Tensor) and
                 audio_query.numel()
                 else (tab_features.shape[0]
                       if isinstance(tab_features, torch.Tensor)
                       else (len(textual_query)
                             if isinstance(textual_query,
                                           (list, tuple))
                             else 1)))
            img_pooled = torch.zeros(b, self.embed_dim,
                                     device=device)

        use_text = (self.base_use_text and
                    (not self.training or
                     random.random() > self.p_drop_text))
        use_tab = (self.base_use_tab and
                   tab_features is not None and
                   tab_features.numel() > 0 and
                   (not self.training or
                    random.random() > self.p_drop_tab))
        use_audio = (self.base_use_audio and
                     audio_query is not None and
                     audio_query.numel() > 0 and
                     (not self.training or
                      random.random() > self.p_drop_audio))

        txt_feats = (self._encode_text_raw(textual_query, device)
                     if use_text else None)
        tab_feats = (self.tab_embed(
            tab_features.to(device, non_blocking=True))
                     if use_tab else None)
        aud_feats = (self.extract_audio_fea(
            audio_query, audio_mask, device)
                     if use_audio else None)

        img_for_clip = F.normalize(img_pooled, dim=-1)

        xattn_weights = None
        if (self.use_cross_attn and
                any(x is not None
                    for x in (txt_feats, tab_feats, aud_feats))):
            ctx_list = [x for x in (txt_feats, tab_feats, aud_feats)
                        if x is not None]
            img_refined, xattn_weights = self.cross_attention(
                img_pooled.unsqueeze(1),
                torch.stack(ctx_list, dim=1),
                return_weights=return_attn_weights)
            img_refined = img_refined.squeeze(1)
        else:
            img_refined = img_pooled

        modalities = [img_refined, tab_feats, aud_feats]
        fused = sanitize_tensor(self.modality_fusion(modalities), clamp=1e4)

        x = self.fusion2(self.fusion1(fused))
        x = sanitize_tensor(x, clamp=1e4)
        logits = sanitize_tensor(self.classifier(x).squeeze(1), clamp=30.0)

        if return_embeds and return_attn_weights:
            embeds = {"img": img_for_clip, "txt": txt_feats,
                      "aud": aud_feats, "tab": tab_feats,
                      "fused": F.normalize(fused, dim=-1)}
            return logits, embeds, xattn_weights
        if return_embeds:
            return logits, {
                "img": img_for_clip, "txt": txt_feats,
                "aud": aud_feats, "tab": tab_feats,
                "fused": F.normalize(fused, dim=-1)}
        if return_attn_weights:
            return logits, xattn_weights
        return logits

    def _pair_nce(self, fa, fb):
        if fa is None or fb is None:
            return torch.tensor(0.0,
                                device=self.logit_scale.device)
        mask = ((fa.abs().sum(dim=1) > 0) &
                (fb.abs().sum(dim=1) > 0))
        if mask.sum() < 2:
            return torch.tensor(0.0,
                                device=self.logit_scale.device)
        fa, fb = fa[mask], fb[mask]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (fa @ fb.t())
        labels = torch.arange(fa.size(0), device=fa.device)
        return 0.5 * (F.cross_entropy(logits, labels) +
                      F.cross_entropy(logits.t(), labels))

    def audioclip_joint_loss(self, embeds):
        terms = []
        for a, b in [("img", "txt"), ("txt", "aud"), ("img", "aud")]:
            loss = self._pair_nce(embeds.get(a), embeds.get(b))
            if (torch.is_tensor(loss) and
                    torch.isfinite(loss) and
                    loss.requires_grad):
                terms.append(loss)
        return (torch.stack(terms).mean() if terms
                else torch.tensor(0.0,
                                  device=self.logit_scale.device))


# ═══════════════════════════════════════════════════════════════════════
# Training / evaluation utilities
# ═══════════════════════════════════════════════════════════════════════
def sanitize_tensor(x: torch.Tensor, clamp: float = 30.0) -> torch.Tensor:
    if not torch.is_tensor(x):
        return x
    return torch.nan_to_num(x.float(), nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


def sanitize_prob_array(x, default: float = 0.5):
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=default, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def safe_binary_auc(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = sanitize_prob_array(y_prob)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y, p))
    except Exception as e:
        logging.warning("AUC failed; returning 0.0. Reason: %s", e)
        return 0.0


def safe_binary_ap(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = sanitize_prob_array(y_prob)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y, p))
    except Exception as e:
        logging.warning("AP failed; returning 0.0. Reason: %s", e)
        return 0.0


def evaluate(model, dataloader, device, criterion,
             threshold=0.5, use_tta=False, tta_runs=2,
             return_attn_weights=False):
    model.eval()
    all_probs, all_labels, all_logits = [], [], []
    losses, all_ids, all_attn_w = [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            vq = batch["visual_query"].to(device, non_blocking=True)
            im = batch["image_mask"].to(device, non_blocking=True)
            aq = batch.get("audio_query")
            am = batch.get("audio_mask")
            if isinstance(aq, torch.Tensor):
                aq = aq.to(device, non_blocking=True)
            if isinstance(am, torch.Tensor):
                am = am.to(device, non_blocking=True)
            tq = batch["textual_query"]
            tf = batch.get("tab_features")
            if tf is not None:
                tf = tf.to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with amp_autocast(device):
                if return_attn_weights:
                    out = model(tq, vq, im, tf,
                                audio_query=aq, audio_mask=am,
                                return_embeds=False,
                                return_attn_weights=True)
                    if isinstance(out, tuple):
                        z, attn_w = out[0], out[-1]
                    else:
                        z, attn_w = out, None
                    if attn_w is not None:
                        all_attn_w.append(attn_w.detach().cpu())
                else:
                    z = model(tq, vq, im, tf,
                              audio_query=aq, audio_mask=am,
                              return_embeds=False)
                z = sanitize_tensor(z, clamp=30.0)
                loss = criterion(z, labels)
                if not torch.isfinite(loss):
                    logging.warning("Non-finite evaluation loss; replacing with 0.0")
                    loss = torch.zeros((), device=device)
            probs = torch.sigmoid(z).clamp(0.0, 1.0)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            losses.append(float(loss.item()))
            all_logits.extend(z.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            pids = batch.get("patient_id")
            if isinstance(pids, torch.Tensor):
                all_ids.extend(
                    pids.detach().cpu().numpy().astype(int).tolist())

    if not all_labels:
        empty = {k: 0.0 for k in
                 ["loss", "accuracy", "balanced_accuracy",
                  "precision", "recall", "f1", "auc", "ap",
                  "brier", "ece"]}
        empty["loss"] = float(np.mean(losses)) if losses else 0.0
        return empty, np.array([]), np.array([]), np.array([]), \
            np.array([]), all_attn_w

    y = np.asarray(all_labels, dtype=int)
    p = sanitize_prob_array(all_probs).astype(np.float32)
    lg = np.nan_to_num(np.asarray(all_logits, dtype=np.float32), nan=0.0, posinf=30.0, neginf=-30.0)
    yhat = (p >= threshold).astype(int)

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "auc": safe_binary_auc(y, p),
        "ap": safe_binary_ap(y, p),
        "brier": float(brier_score(y, p)),
        "ece": float(ece_score(y, p)),
    }
    return (metrics, p, y, lg,
            np.asarray(all_ids, dtype=np.int64), all_attn_w)


def find_best_threshold(y_true, y_prob,
                        optimize_for="balanced_accuracy",
                        min_threshold=0.25,
                        max_threshold=0.75):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(
        np.nan_to_num(np.asarray(y_prob, dtype=float),
                      nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if y_prob.size == 0:
        return 0.5, {"threshold": 0.5, "f1": 0.0,
                     "accuracy": 0.0, "balanced_accuracy": 0.0}
    if len(np.unique(y_true)) < 2:
        return 0.5, {
            "threshold": 0.5,
            "f1": float(f1_score(y_true,
                                  (y_prob >= 0.5).astype(int),
                                  zero_division=0))}
    best_t, best_val, best_metrics = 0.5, -1.0, {}
    _, _, thresh = precision_recall_curve(y_true, y_prob)
    lo = float(np.clip(min_threshold, 0.0, 1.0))
    hi = float(np.clip(max_threshold, lo + 1e-6, 1.0))
    candidates = np.unique(np.concatenate(
        [thresh, np.linspace(lo, hi, 51), [0.5]]))
    candidates = candidates[
        np.isfinite(candidates) &
        (candidates >= lo) & (candidates <= hi)]
    if candidates.size == 0:
        candidates = np.array([0.5])
    # AUC is threshold-independent; compute once (was recomputed per candidate).
    auc = safe_binary_auc(y_true, y_prob)
    for t in candidates:
        yhat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        acc = accuracy_score(y_true, yhat)
        bacc = balanced_accuracy_score(y_true, yhat)
        rec = recall_score(y_true, yhat, zero_division=0)
        val = {"f1": f1, "accuracy": acc,
               "balanced_accuracy": bacc,
               "auc": auc, "recall": rec}.get(optimize_for, f1)
        if val > best_val:
            best_val, best_t = val, float(t)
            best_metrics = {
                "threshold": best_t, "f1": f1, "accuracy": acc,
                "balanced_accuracy": bacc, "auc": auc, "recall": rec}
    return best_t, best_metrics


def temperature_scaling(logits, labels, max_iter=100):
    if logits.size == 0 or len(np.unique(labels)) < 2:
        return 1.0
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logits = np.nan_to_num(np.asarray(logits, dtype=np.float32), nan=0.0, posinf=30.0, neginf=-30.0)
    labels = np.asarray(labels).astype(np.float32)
    z = torch.tensor(logits, dtype=torch.float32, device=device).clamp(-30.0, 30.0)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    temp = nn.Parameter(torch.ones([], device=device))
    opt = torch.optim.LBFGS([temp], lr=0.1, max_iter=50,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            z / (temp.abs() + 1e-6), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass
    return float(np.clip(
        float(temp.abs().detach().cpu().item()), 0.75, 4.0))


def apply_temperature_to_logits(logits, temp):
    lg = np.nan_to_num(np.asarray(logits, dtype=np.float64), nan=0.0, posinf=30.0, neginf=-30.0)
    lg = np.clip(lg, -30.0, 30.0)
    t = float(np.clip(temp, 0.75, 4.0))
    return sanitize_prob_array(1.0 / (1.0 + np.exp(-lg / t)))


def brier_score(y_true, y_prob):
    y = np.asarray(y_true, dtype=np.float64)
    p = sanitize_prob_array(y_prob)
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def ece_score(y_true, y_prob, n_bins=15):
    y = np.asarray(y_true, dtype=np.int32)
    p = sanitize_prob_array(y_prob).astype(np.float32)
    if p.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = ((p >= lo) &
             ((p < hi) if i < n_bins - 1 else (p <= hi)))
        if m.any():
            ece += float(m.mean()) * abs(
                float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def isotonic_calibration_fit(probs, labels):
    p = sanitize_prob_array(probs)
    y = np.asarray(labels).astype(int)
    if p.size < 10 or len(np.unique(y)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    return ir


def make_loader(dataset, batch_size, shuffle=None, sampler=None,
                num_workers=0, pin_memory=True, drop_last=False):
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, drop_last=drop_last,
                  worker_init_fn=worker_init_fn)
    if sampler is not None:
        kwargs["sampler"] = sampler
    elif shuffle is not None:
        kwargs["shuffle"] = shuffle
    if num_workers and num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def train_one_epoch(model, optimizer, dataloader, scaler,
                    device, criterion,
                    label_smoothing=0.07,
                    use_label_smoothing=True,
                    log_every=50, ema=None,
                    clip_weight=0.02,
                    accumulation_steps=1):
    model.train()
    loss_avg = RunningAverage()
    acc_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    has_unflushed = False
    for i, batch in enumerate(dataloader, 1):
        vq = batch["visual_query"].to(device, non_blocking=True)
        im = batch["image_mask"].to(device, non_blocking=True)
        aq = batch.get("audio_query")
        am = batch.get("audio_mask")
        if isinstance(aq, torch.Tensor):
            aq = aq.to(device, non_blocking=True)
        if isinstance(am, torch.Tensor):
            am = am.to(device, non_blocking=True)
        tq = batch["textual_query"]
        tf = batch.get("tab_features")
        if tf is not None:
            tf = tf.to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with amp_autocast(device):
            if clip_weight > 0:
                logits, embeds = model(
                    tq, vq, im, tf,
                    audio_query=aq, audio_mask=am,
                    return_embeds=True)
            else:
                logits = model(tq, vq, im, tf,
                               audio_query=aq, audio_mask=am,
                               return_embeds=False)
                embeds = None
            targets = (smooth_labels(labels, eps=label_smoothing)
                       if use_label_smoothing else labels)
            loss_cls = criterion(logits, targets)
            loss_clip = (model.audioclip_joint_loss(embeds)
                         if clip_weight > 0 and embeds is not None
                         else torch.tensor(0.0, device=device))
            unscaled = loss_cls + clip_weight * loss_clip
            if not torch.isfinite(unscaled):
                logging.warning("Skipping non-finite training batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = unscaled / acc_steps
        scaler.scale(loss).backward()
        has_unflushed = True
        if (i % acc_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            has_unflushed = False
            if ema is not None:
                ema.update(model)
        loss_avg.update(float(unscaled.detach().item()))
        if log_every > 0 and i % log_every == 0:
            logging.info("  Step %d | loss %.4f", i, loss_avg())
    if has_unflushed:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    return loss_avg()


def _safe_logit(p, eps=1e-6):
    return float(np.log(
        np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))))


def set_classifier_prior_bias(model, pos_prior):
    try:
        target = (model._orig_mod
                  if hasattr(model, "_orig_mod") else model)
        last = target.classifier[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.bias.data.fill_(_safe_logit(pos_prior))
    except Exception as e:
        logging.warning("Failed to set prior bias: %s", e)


def count_trainable_parameters(model) -> int:
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    return int(sum(p.numel() for p in target.parameters() if p.requires_grad))


def reset_optimizer_after_unfreeze(model, base_lr: float, weight_decay: float):
    decay_p, no_decay_p = [], []
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, p in target.named_parameters():
        if not p.requires_grad:
            continue
        if (name.endswith(".bias") or any(
                k in name.lower()
                for k in ("layernorm", "layer_norm", "bn", "logit_scale"))):
            no_decay_p.append(p)
        else:
            decay_p.append(p)
    return optim.AdamW(
        [{"params": decay_p, "lr": base_lr, "weight_decay": weight_decay},
         {"params": no_decay_p, "lr": base_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.98), eps=1e-6)


def validate_tabular_integration(train_ds, expected_tab_dim: int):
    logging.info("Validating tabular integration...")
    sample = train_ds[0]
    assert "tab_features" in sample, "tab_features key missing"
    tab_tensor = sample["tab_features"]
    assert tab_tensor.shape[0] == expected_tab_dim, \
        f"Dim mismatch: expected {expected_tab_dim}, got {tab_tensor.shape[0]}"
    logging.info("Tabular features: present, correct shape")


def _filter_modality_map_by_df(modality_map: Dict[int, List[str]],
                               df: pd.DataFrame) -> Dict[int, List[str]]:
    allowed = set(df["Patient number"].astype(int).tolist())
    return {int(pid): list(modality_map.get(int(pid), [])) for pid in allowed}


# ═══════════════════════════════════════════════════════════════════════
# Reporting, XAI, ablation, robustness utilities
# ═══════════════════════════════════════════════════════════════════════
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_binary_metrics(y_true, y_prob, threshold=0.5, n_bins=15):
    y = np.asarray(y_true, dtype=int)
    p = sanitize_prob_array(y_prob).astype(np.float32)
    yhat = (p >= float(threshold)).astype(int)
    if y.size == 0:
        return {k: 0.0 for k in [
            "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "auc", "ap", "brier", "ece"]}
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "auc": safe_binary_auc(y, p),
        "ap": safe_binary_ap(y, p),
        "brier": float(brier_score(y, p)),
        "ece": float(ece_score(y, p, n_bins=n_bins)),
        "threshold": float(threshold),
    }


def validation_selection_score(metrics: Dict[str, float], probs=None) -> float:
    auc = float(metrics.get("auc", 0.0))
    bacc = float(metrics.get("balanced_accuracy", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    brier = float(metrics.get("brier", 0.25))
    ece = float(metrics.get("ece", 0.25))
    score = 0.50 * auc + 0.25 * bacc + 0.15 * f1 - 0.07 * brier - 0.03 * ece
    if probs is not None:
        pp = sanitize_prob_array(probs)
        if pp.size and float(np.std(pp)) < 0.01:
            score -= 0.25
    return float(score)


def is_collapsed_prediction(y_prob, auc_value=None) -> bool:
    p = sanitize_prob_array(y_prob)
    if p.size == 0:
        return False
    low_var = float(np.std(p)) < 0.005
    low_auc = auc_value is not None and float(auc_value) <= 0.52
    return bool(low_var and low_auc)


def save_predictions_csv(patient_ids, labels, probs, threshold, save_path):
    ensure_dir(os.path.dirname(save_path) or ".")
    y = np.asarray(labels, dtype=int)
    p = sanitize_prob_array(probs)
    ids = np.asarray(patient_ids if len(patient_ids) else np.arange(len(y)))
    df = pd.DataFrame({
        "patient_id": ids.astype(int) if len(ids) == len(y) else np.arange(len(y)),
        "label": y,
        "probability": p,
        "prediction": (p >= float(threshold)).astype(int),
    })
    df.to_csv(save_path, index=False)


def save_metrics_json(metrics: Dict[str, Any], save_path: str):
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    save_dict_to_json(clean, save_path)


def plot_training_history(history: List[Dict[str, float]], save_dir: str, prefix: str):
    ensure_dir(save_dir)
    if not history:
        return
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, f"{prefix}_training_history.csv"), index=False)
    plt.close("all")
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Training loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training and validation loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_train_val_loss.png")); plt.close()
    metric_cols = [c for c in ["val_auc", "val_f1", "val_recall", "val_balanced_accuracy"] if c in df]
    if metric_cols:
        plt.figure()
        for c in metric_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch"); plt.ylabel("Metric")
        plt.title("Validation metrics"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_validation_metrics.png")); plt.close()


def plot_confusion_matrix_png(y_true, y_prob, threshold, save_path, title="Confusion matrix"):
    ensure_dir(os.path.dirname(save_path) or ".")
    y = np.asarray(y_true, dtype=int)
    p = sanitize_prob_array(y_prob)
    yhat = (p >= float(threshold)).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    plt.close("all")
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(save_path); plt.close()
    return cm


def plot_probability_distribution(y_true, y_prob, save_path, title="Probability distribution"):
    ensure_dir(os.path.dirname(save_path) or ".")
    y = np.asarray(y_true, dtype=int)
    p = sanitize_prob_array(y_prob)
    plt.close("all")
    plt.figure()
    if len(p[y == 0]):
        plt.hist(p[y == 0], bins=15, alpha=0.65, label="Class 0")
    if len(p[y == 1]):
        plt.hist(p[y == 1], bins=15, alpha=0.65, label="Class 1")
    plt.xlabel("Predicted probability"); plt.ylabel("Count")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(save_path); plt.close()


def plot_calibration_curve(y_true, y_prob, save_path, n_bins=15, title="Calibration plot"):
    ensure_dir(os.path.dirname(save_path) or ".")
    y = np.asarray(y_true, dtype=int)
    p = sanitize_prob_array(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if m.any():
            xs.append(float(p[m].mean())); ys.append(float(y[m].mean())); counts.append(int(m.sum()))
    plt.close("all")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    if xs:
        plt.plot(xs, ys, marker="o", label="Model")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed positive frequency")
    plt.title(f"{title}\nBrier={brier_score(y, p):.4f}, ECE={ece_score(y, p, n_bins=n_bins):.4f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path); plt.close()
    pd.DataFrame({"mean_predicted_probability": xs,
                  "observed_positive_frequency": ys,
                  "count": counts}).to_csv(save_path.replace(".png", ".csv"), index=False)


def save_standard_eval_artifacts(y_true, y_prob, patient_ids, threshold, save_dir, prefix, n_bins=15):
    ensure_dir(save_dir)
    metrics = compute_binary_metrics(y_true, y_prob, threshold, n_bins=n_bins)
    save_metrics_json(metrics, os.path.join(save_dir, f"{prefix}_metrics.json"))
    save_predictions_csv(patient_ids, y_true, y_prob, threshold,
                         os.path.join(save_dir, f"{prefix}_predictions.csv"))
    plot_confusion_matrix_png(y_true, y_prob, threshold,
                              os.path.join(save_dir, f"{prefix}_confusion_matrix.png"),
                              title=f"{prefix} confusion matrix")
    plot_probability_distribution(y_true, y_prob,
                                  os.path.join(save_dir, f"{prefix}_probability_distribution.png"),
                                  title=f"{prefix} probability distribution")
    plot_roc_pr(y_true, y_prob, save_dir, prefix)
    plot_calibration_curve(y_true, y_prob,
                           os.path.join(save_dir, f"{prefix}_calibration.png"),
                           n_bins=n_bins, title=f"{prefix} calibration")
    return metrics


def _temporarily_set_model_flags(model, use_text=None, use_tab=None, use_audio=None):
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    old = (target.base_use_text, target.base_use_tab, target.base_use_audio)
    if use_text is not None:
        target.base_use_text = bool(use_text)
    if use_tab is not None:
        target.base_use_tab = bool(use_tab)
    if use_audio is not None:
        target.base_use_audio = bool(use_audio)
    return target, old


def _restore_model_flags(target, old):
    target.base_use_text, target.base_use_tab, target.base_use_audio = old


def _apply_batch_modality_config(batch: Dict[str, Any], config: Dict[str, bool]):
    out = dict(batch)
    if not config.get("image", True):
        if isinstance(out.get("visual_query"), torch.Tensor):
            out["visual_query"] = torch.zeros_like(out["visual_query"])
        if isinstance(out.get("image_mask"), torch.Tensor):
            out["image_mask"] = torch.zeros_like(out["image_mask"])
    if not config.get("tabular", True):
        if isinstance(out.get("tab_features"), torch.Tensor):
            out["tab_features"] = torch.zeros_like(out["tab_features"])
    if not config.get("audio", True):
        if isinstance(out.get("audio_query"), torch.Tensor):
            out["audio_query"] = torch.zeros_like(out["audio_query"])
        if isinstance(out.get("audio_mask"), torch.Tensor):
            out["audio_mask"] = torch.zeros_like(out["audio_mask"])
    if not config.get("text", True):
        tq = out.get("textual_query", [])
        if isinstance(tq, (list, tuple)):
            out["textual_query"] = ["" for _ in tq]
        else:
            out["textual_query"] = ""
    return out


def evaluate_under_modality_config(model, dataloader, device, criterion,
                                   config: Dict[str, bool], threshold=0.5,
                                   temp=1.0, iso_reg=None, n_bins=15):
    target, old = _temporarily_set_model_flags(
        model,
        use_text=config.get("text", True),
        use_tab=config.get("tabular", True),
        use_audio=config.get("audio", True))
    target.eval()
    losses, logits, probs, labels, ids = [], [], [], [], []
    try:
        with torch.inference_mode():
            for batch in dataloader:
                batch = _apply_batch_modality_config(batch, config)
                vq = batch["visual_query"].to(device, non_blocking=True)
                im = batch["image_mask"].to(device, non_blocking=True)
                aq = batch.get("audio_query")
                am = batch.get("audio_mask")
                if isinstance(aq, torch.Tensor):
                    aq = aq.to(device, non_blocking=True)
                if isinstance(am, torch.Tensor):
                    am = am.to(device, non_blocking=True)
                tf = batch.get("tab_features")
                if tf is not None:
                    tf = tf.to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with amp_autocast(device):
                    z = target(batch["textual_query"], vq, im, tf,
                               audio_query=aq, audio_mask=am,
                               return_embeds=False)
                    z = sanitize_tensor(z, clamp=30.0)
                    loss = criterion(z, y)
                losses.append(float(loss.item()) if torch.isfinite(loss) else 0.0)
                logits.extend(z.detach().cpu().tolist())
                probs.extend(torch.sigmoid(z).detach().cpu().tolist())
                labels.extend(y.detach().cpu().tolist())
                pids = batch.get("patient_id")
                if isinstance(pids, torch.Tensor):
                    ids.extend(pids.detach().cpu().numpy().astype(int).tolist())
    finally:
        _restore_model_flags(target, old)
    lg = np.asarray(logits, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    p_cal = apply_temperature_to_logits(lg, temp)
    if iso_reg is not None and len(p_cal):
        p_cal = sanitize_prob_array(iso_reg.predict(p_cal))
    metrics = compute_binary_metrics(y, p_cal, threshold=threshold, n_bins=n_bins)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, p_cal, y, lg, np.asarray(ids, dtype=np.int64)


def modality_ablation_configs(use_audio=True):
    return {
        "full": {"image": True, "text": True, "tabular": True, "audio": bool(use_audio)},
        "image_only": {"image": True, "text": False, "tabular": False, "audio": False},
        "clinical_only_text_tabular": {"image": False, "text": True, "tabular": True, "audio": False},
        "no_image": {"image": False, "text": True, "tabular": True, "audio": bool(use_audio)},
        "no_text": {"image": True, "text": False, "tabular": True, "audio": bool(use_audio)},
        "no_tabular": {"image": True, "text": True, "tabular": False, "audio": bool(use_audio)},
        "no_audio": {"image": True, "text": True, "tabular": True, "audio": False},
    }


def run_modality_ablation_suite(model, dataloader, device, criterion,
                                temp, iso_reg, threshold, save_dir, split_name,
                                use_audio=True, n_bins=15):
    ensure_dir(save_dir)
    rows = []
    for name, cfg in modality_ablation_configs(use_audio=use_audio).items():
        m, p, y, lg, ids = evaluate_under_modality_config(
            model, dataloader, device, criterion, cfg, threshold=threshold,
            temp=temp, iso_reg=iso_reg, n_bins=n_bins)
        row = {"split": split_name, "configuration": name, **m}
        row.update({f"keep_{k}": int(v) for k, v in cfg.items()})
        rows.append(row)
        save_predictions_csv(ids, y, p, threshold,
                             os.path.join(save_dir, f"{split_name}_{name}_predictions.csv"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{split_name}_modality_ablation_7configs.csv"), index=False)
    save_dict_to_json({"rows": rows}, os.path.join(save_dir, f"{split_name}_modality_ablation_7configs.json"))
    metric = "auc" if "auc" in df.columns else "f1"
    plt.close("all")
    plt.figure(figsize=(9, 4.5))
    plt.bar(df["configuration"].astype(str), df[metric].astype(float))
    plt.xticks(rotation=35, ha="right"); plt.ylabel(metric.upper())
    plt.title(f"{split_name} modality ablation: {metric.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_modality_ablation_{metric}.png")); plt.close()
    return rows


def run_missing_modality_robustness(model, dataloader, device, criterion,
                                    temp, iso_reg, threshold, save_dir, split_name,
                                    use_audio=True, n_bins=15):
    rows = run_modality_ablation_suite(
        model, dataloader, device, criterion, temp, iso_reg, threshold,
        save_dir, f"{split_name}_missing_robustness", use_audio=use_audio, n_bins=n_bins)
    return rows


def _active_context_names(model):
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    names = []
    if getattr(target, "base_use_text", False):
        names.append("text")
    if getattr(target, "base_use_tab", False):
        names.append("tabular")
    if getattr(target, "base_use_audio", False):
        names.append("audio")
    return names


def collect_and_save_attention_weights(model, dataloader, device, save_dir, split_name, max_batches=None):
    ensure_dir(save_dir)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.eval()
    context_names = _active_context_names(target)
    rows = []
    with torch.inference_mode():
        for bi, batch in enumerate(dataloader):
            if max_batches is not None and bi >= max_batches:
                break
            vq = batch["visual_query"].to(device, non_blocking=True)
            im = batch["image_mask"].to(device, non_blocking=True)
            aq = batch.get("audio_query")
            am = batch.get("audio_mask")
            if isinstance(aq, torch.Tensor):
                aq = aq.to(device, non_blocking=True)
            if isinstance(am, torch.Tensor):
                am = am.to(device, non_blocking=True)
            tf = batch.get("tab_features")
            if tf is not None:
                tf = tf.to(device, non_blocking=True)
            with amp_autocast(device):
                out = target(batch["textual_query"], vq, im, tf,
                             audio_query=aq, audio_mask=am,
                             return_embeds=False, return_attn_weights=True)
            if not isinstance(out, tuple) or out[-1] is None:
                continue
            attn = out[-1].detach().cpu().float().squeeze(1).numpy()
            pids = batch.get("patient_id")
            if isinstance(pids, torch.Tensor):
                pids = pids.detach().cpu().numpy().astype(int).tolist()
            else:
                pids = list(range(len(attn)))
            for r, pid in zip(attn, pids):
                row = {"patient_id": int(pid)}
                for i, val in enumerate(r):
                    key = context_names[i] if i < len(context_names) else f"context_{i}"
                    row[f"attention_to_{key}"] = float(val)
                rows.append(row)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{split_name}_cross_modal_attention_weights.csv"), index=False)
    means = {c: float(df[c].mean()) for c in df.columns if c.startswith("attention_to_")}
    save_dict_to_json(means, os.path.join(save_dir, f"{split_name}_cross_modal_attention_summary.json"))
    if means:
        plt.close("all")
        plt.figure(figsize=(6.5, 4.0))
        plt.bar(list(means.keys()), list(means.values()))
        plt.xticks(rotation=30, ha="right"); plt.ylabel("Mean attention weight")
        plt.title(f"{split_name} cross-modal attention"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_cross_modal_attention_weights.png")); plt.close()
    return means


def _to_uint8_img(t: torch.Tensor):
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return (arr * 255).astype(np.uint8)


def save_gradcam_heatmaps(model, dataloader, device, save_dir, split_name,
                          max_samples=8,
                          target_patient_ids: Optional[List[int]] = None,
                          make_composite: bool = True):
    """
    Grad-CAM-style input-gradient heatmaps for X-ray samples.

    In addition to individual patient heatmaps, this creates a journal-ready
    composite for selected patients. For the requested patients 16, 20, 22,
    and 24, the composite is arranged as a 2 x 4 grid:

        row 1: P16 original | P16 heatmap | P20 original | P20 heatmap
        row 2: P22 original | P22 heatmap | P24 original | P24 heatmap
    """
    ensure_dir(save_dir)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    old_freeze_flag = getattr(target, "freeze_biomedclip", False)
    target.freeze_biomedclip = False
    target.eval()

    requested = ([int(x) for x in target_patient_ids]
                 if target_patient_ids else [])
    requested_set = set(requested)
    composite_items: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    saved = 0
    rows = []

    def _should_keep(pid: int) -> bool:
        if requested_set:
            return int(pid) in requested_set
        return saved < int(max_samples)

    try:
        for batch in dataloader:
            if requested_set and requested_set.issubset(set(composite_items.keys())):
                break
            if not requested_set and saved >= int(max_samples):
                break
            if not isinstance(batch.get("visual_query"), torch.Tensor):
                continue
            vq0 = batch["visual_query"]
            if vq0.dim() != 5:
                continue
            vq = vq0.to(device).clone().detach().requires_grad_(True)
            im = batch["image_mask"].to(device)
            aq = batch.get("audio_query"); am = batch.get("audio_mask")
            if isinstance(aq, torch.Tensor):
                aq = aq.to(device)
            if isinstance(am, torch.Tensor):
                am = am.to(device)
            tf = batch.get("tab_features")
            if tf is not None:
                tf = tf.to(device)
            target.zero_grad(set_to_none=True)
            with safe_no_amp_context(device):
                logits = target(batch["textual_query"], vq, im, tf,
                                audio_query=aq, audio_mask=am,
                                return_embeds=False)
                # Use positive-class evidence for clearer saliency.
                score = logits.sum()
            score.backward()
            grad = vq.grad.detach().abs().mean(dim=2)
            pids = batch.get("patient_id")
            if isinstance(pids, torch.Tensor):
                pids = pids.detach().cpu().numpy().astype(int).tolist()
            else:
                pids = list(range(vq.shape[0]))

            for bi in range(vq.shape[0]):
                pid = int(pids[bi])
                if not _should_keep(pid):
                    continue
                valid_imgs = torch.where(im[bi].detach().cpu() > 0.5)[0]
                if len(valid_imgs) == 0:
                    continue
                ki = int(valid_imgs[0].item())
                heat = grad[bi, ki]
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                img = _to_uint8_img(vq[bi, ki])
                heat_np = heat.detach().cpu().numpy()

                plt.close("all")
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img); plt.axis("off"); plt.title(f"Patient {pid}: original")
                plt.subplot(1, 2, 2)
                plt.imshow(img); plt.imshow(heat_np, alpha=0.45, cmap="jet")
                plt.axis("off"); plt.title(f"Patient {pid}: Grad-CAM")
                plt.tight_layout()
                fname = f"{split_name}_patient_{pid}_img{ki}_gradcam.png"
                path = os.path.join(save_dir, fname)
                plt.savefig(path, dpi=180); plt.close()
                rows.append({"patient_id": pid, "image_index": ki, "path": path})

                if requested_set and pid in requested_set:
                    composite_items[pid] = (img, heat_np)
                saved += 1

        if requested:
            missing = [pid for pid in requested if pid not in composite_items]
            if missing:
                logging.warning(
                    "Grad-CAM composite missing requested patient(s): %s. "
                    "They were not present in the supplied dataloader.", missing)

        if make_composite and requested and composite_items:
            fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
            for ax in axes.ravel():
                ax.axis("off")
            for j, pid in enumerate(requested[:4]):
                if pid not in composite_items:
                    continue
                row = j // 2
                col = (j % 2) * 2
                img, heat_np = composite_items[pid]
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Patient {pid}\nOriginal", fontsize=12)
                axes[row, col].axis("off")
                axes[row, col + 1].imshow(img)
                axes[row, col + 1].imshow(heat_np, alpha=0.45, cmap="jet")
                axes[row, col + 1].set_title(f"Patient {pid}\nHeatmap", fontsize=12)
                axes[row, col + 1].axis("off")
            fig.suptitle("Grad-CAM composite: original and heatmap pairs", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            comp_name = (f"{split_name}_patients_" +
                         "_".join(str(x) for x in requested[:4]) +
                         "_gradcam_composite_2x4.png")
            comp_path = os.path.join(save_dir, comp_name)
            fig.savefig(comp_path, dpi=220)
            plt.close(fig)
            rows.append({
                "patient_id": "composite",
                "image_index": -1,
                "path": comp_path,
                "patients": ",".join(str(x) for x in requested[:4]),
            })
            logging.info("Saved Grad-CAM composite: %s", comp_path)

    except Exception as e:
        logging.warning("Grad-CAM heatmap generation failed: %s", e)
    finally:
        target.freeze_biomedclip = old_freeze_flag
        target.zero_grad(set_to_none=True)

    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, f"{split_name}_gradcam_index.csv"), index=False)
    return rows


def _batch_logits_for_subset(model, batch, device, subset_cfg):
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    batch2 = _apply_batch_modality_config(batch, subset_cfg)
    old_target, old = _temporarily_set_model_flags(
        target,
        use_text=subset_cfg.get("text", True),
        use_tab=subset_cfg.get("tabular", True),
        use_audio=subset_cfg.get("audio", True))
    try:
        vq = batch2["visual_query"].to(device)
        im = batch2["image_mask"].to(device)
        aq = batch2.get("audio_query"); am = batch2.get("audio_mask")
        if isinstance(aq, torch.Tensor):
            aq = aq.to(device)
        if isinstance(am, torch.Tensor):
            am = am.to(device)
        tf = batch2.get("tab_features")
        if tf is not None:
            tf = tf.to(device)
        with torch.inference_mode():
            with amp_autocast(device):
                z = target(batch2["textual_query"], vq, im, tf,
                           audio_query=aq, audio_mask=am, return_embeds=False)
        return torch.sigmoid(sanitize_tensor(z)).detach().cpu().numpy()
    finally:
        _restore_model_flags(old_target, old)


def save_modality_shap_values(model, dataloader, device, save_dir, split_name, max_samples=8):
    """Exact Shapley-value modality attribution over image/text/tabular/audio coalitions."""
    ensure_dir(save_dir)
    modalities = ["image", "text", "tabular", "audio"]
    rows = []
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.eval()
    for batch in dataloader:
        if len(rows) >= int(max_samples):
            break
        bsz = int(batch["label"].shape[0])
        need = int(max_samples) - len(rows)
        coal_probs = {}
        for r in range(len(modalities) + 1):
            for comb in combinations(modalities, r):
                key = tuple(sorted(comb))
                cfg = {m: (m in key) for m in modalities}
                coal_probs[key] = _batch_logits_for_subset(target, batch, device, cfg)
        pids = batch.get("patient_id")
        if isinstance(pids, torch.Tensor):
            pids = pids.detach().cpu().numpy().astype(int).tolist()
        else:
            pids = list(range(bsz))
        for i in range(min(bsz, need)):
            row = {"patient_id": int(pids[i]), "base_probability": float(coal_probs[tuple()][i]),
                   "full_probability": float(coal_probs[tuple(sorted(modalities))][i])}
            for m in modalities:
                others = [x for x in modalities if x != m]
                phi = 0.0
                for r in range(len(others) + 1):
                    for s in combinations(others, r):
                        S = tuple(sorted(s))
                        S_m = tuple(sorted(list(s) + [m]))
                        weight = (math.factorial(len(S)) *
                                  math.factorial(len(modalities) - len(S) - 1) /
                                  math.factorial(len(modalities)))
                        phi += weight * (float(coal_probs[S_m][i]) - float(coal_probs[S][i]))
                row[f"shap_{m}"] = float(phi)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{split_name}_modality_shap_values.csv"), index=False)
    if not df.empty:
        means = {c: float(df[c].abs().mean()) for c in df.columns if c.startswith("shap_")}
        save_dict_to_json(means, os.path.join(save_dir, f"{split_name}_modality_shap_summary.json"))
        plt.close("all")
        plt.figure(figsize=(6.5, 4.0))
        plt.bar(list(means.keys()), list(means.values()))
        plt.xticks(rotation=30, ha="right"); plt.ylabel("Mean absolute SHAP value")
        plt.title(f"{split_name} modality SHAP summary"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_modality_shap_summary.png")); plt.close()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# NEW (W1): Modality coverage table
# Reports exactly how many patients have X-ray only / X-ray + clinical /
# all three modalities. Saved to results_multimodel.
# ═══════════════════════════════════════════════════════════════════════
def save_modality_coverage_table(df_all, patient_images, patient_audio,
                                 save_dir, use_audio=True):
    ensure_dir(save_dir)
    pids = df_all["Patient number"].astype(int).tolist()
    n = len(pids)
    has_clin = {pid: True for pid in pids}  # tabular/clinical present for all rows
    has_img = {pid: len(patient_images.get(int(pid), [])) > 0 for pid in pids}
    has_aud = {pid: (use_audio and len(patient_audio.get(int(pid), [])) > 0) for pid in pids}

    xray_only = sum(1 for pid in pids
                    if has_img[pid] and not has_aud[pid])
    xray_clin = sum(1 for pid in pids if has_img[pid])  # all rows have clinical
    all_three = sum(1 for pid in pids
                    if has_img[pid] and has_aud[pid] and has_clin[pid])
    clin_only = sum(1 for pid in pids if not has_img[pid] and not has_aud[pid])
    n_img = sum(1 for pid in pids if has_img[pid])
    n_aud = sum(1 for pid in pids if has_aud[pid])

    rows = [
        {"coverage_group": "Total patients", "count": n,
         "percent": 100.0},
        {"coverage_group": "Has clinical (tabular+text)", "count": n,
         "percent": 100.0},
        {"coverage_group": "Has X-ray", "count": n_img,
         "percent": round(100.0 * n_img / max(1, n), 1)},
        {"coverage_group": "Has respiratory audio", "count": n_aud,
         "percent": round(100.0 * n_aud / max(1, n), 1)},
        {"coverage_group": "Clinical only (no X-ray, no audio)", "count": clin_only,
         "percent": round(100.0 * clin_only / max(1, n), 1)},
        {"coverage_group": "X-ray + clinical (no audio)", "count": xray_only,
         "percent": round(100.0 * xray_only / max(1, n), 1)},
        {"coverage_group": "X-ray + clinical (audio optional)", "count": xray_clin,
         "percent": round(100.0 * xray_clin / max(1, n), 1)},
        {"coverage_group": "All three modalities", "count": all_three,
         "percent": round(100.0 * all_three / max(1, n), 1)},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "modality_coverage_table.csv")
    df.to_csv(csv_path, index=False)
    save_dict_to_json({"rows": rows, "n_patients": n},
                      os.path.join(save_dir, "modality_coverage_table.json"))

    # bar figure
    plt.close("all")
    plt.figure(figsize=(9, 4.5))
    plt.barh(df["coverage_group"].astype(str), df["count"].astype(int))
    plt.xlabel("Patient count"); plt.title("Modality coverage (n=%d)" % n)
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "modality_coverage_table.png"), dpi=160)
    plt.close()
    logging.info("Saved modality coverage table to %s", csv_path)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# NEW: collect per-modality embeddings across a loader
# Used by retrieval (W2) and t-SNE feature distribution.
# ═══════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def collect_embeddings(model, dataloader, device):
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.eval()
    store = {k: [] for k in ["img", "txt", "aud", "tab", "fused"]}
    masks = {k: [] for k in ["img", "txt", "aud", "tab"]}
    labels, ids = [], []
    for batch in dataloader:
        vq = batch["visual_query"].to(device, non_blocking=True)
        im = batch["image_mask"].to(device, non_blocking=True)
        aq = batch.get("audio_query"); am = batch.get("audio_mask")
        if isinstance(aq, torch.Tensor):
            aq = aq.to(device, non_blocking=True)
        if isinstance(am, torch.Tensor):
            am = am.to(device, non_blocking=True)
        tf = batch.get("tab_features")
        if tf is not None:
            tf = tf.to(device, non_blocking=True)
        with amp_autocast(device):
            _, embeds = target(batch["textual_query"], vq, im, tf,
                               audio_query=aq, audio_mask=am,
                               return_embeds=True)
        B = int(batch["label"].shape[0])
        for k in store:
            v = embeds.get(k)
            if v is None:
                store[k].append(np.zeros((B, target.embed_dim), dtype=np.float32))
                if k in masks:
                    masks[k].append(np.zeros((B,), dtype=np.float32))
            else:
                arr = v.detach().cpu().float().numpy()
                store[k].append(arr)
                if k in masks:
                    masks[k].append((np.abs(arr).sum(axis=1) > 0).astype(np.float32))
        labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
        pids = batch.get("patient_id")
        if isinstance(pids, torch.Tensor):
            ids.extend(pids.detach().cpu().numpy().astype(int).tolist())
        else:
            ids.extend(list(range(B)))
    out = {k: (np.concatenate(v, axis=0) if v else np.zeros((0, target.embed_dim), dtype=np.float32))
           for k, v in store.items()}
    out_masks = {k: (np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32))
                 for k, v in masks.items()}
    return out, out_masks, np.asarray(labels, dtype=int), np.asarray(ids, dtype=np.int64)


def _retrieval_metrics_from_sim(sim: np.ndarray, ks=(1, 5, 10)):
    """sim[i, j] = similarity of query i to gallery j. Diagonal = positive."""
    n = sim.shape[0]
    if n == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "n": 0}
    order = np.argsort(-sim, axis=1)
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    res = {}
    for k in ks:
        kk = min(k, n)
        res[f"recall@{k}"] = float(np.mean(ranks <= kk))
    res["mrr"] = float(np.mean(1.0 / ranks))
    res["n"] = int(n)
    return res


def run_cross_modal_retrieval(model, dataloader, device, save_dir, split_name,
                              ks=(1, 5, 10)):
    """
    W2: cross-modal retrieval. Recall@1/5/10 + MRR for the modality pairs
    image<->text (IT) and image<->audio (IA), plus text<->audio. Patients
    that lack a modality are excluded from that pair's gallery (true positive
    matching requires both embeddings present).
    """
    ensure_dir(save_dir)
    emb, masks, labels, ids = collect_embeddings(model, dataloader, device)

    pairs = [("img", "txt", "Image", "Text", "IT"),
             ("img", "aud", "Image", "Audio", "IA"),
             ("txt", "aud", "Text", "Audio", "TA")]
    rows = []
    for a, b, a_name, b_name, tag in pairs:
        ma = masks.get(a); mb = masks.get(b)
        if ma is None or mb is None:
            continue
        keep = (ma > 0.5) & (mb > 0.5)
        if keep.sum() < 2:
            logging.info("Retrieval %s: <2 paired samples, skipping.", tag)
            continue
        ea = emb[a][keep]; eb = emb[b][keep]
        # L2 normalize (defensive; embeddings already normalized)
        ea = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-8)
        eb = eb / (np.linalg.norm(eb, axis=1, keepdims=True) + 1e-8)
        sim_ab = ea @ eb.T   # a (query) -> b (gallery)
        sim_ba = eb @ ea.T   # b (query) -> a (gallery)
        m_ab = _retrieval_metrics_from_sim(sim_ab, ks=ks)
        m_ba = _retrieval_metrics_from_sim(sim_ba, ks=ks)
        rows.append({"pair": tag, "direction": f"{a_name}->{b_name}", **m_ab})
        rows.append({"pair": tag, "direction": f"{b_name}->{a_name}", **m_ba})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, f"{split_name}_cross_modal_retrieval.csv")
    df.to_csv(csv_path, index=False)
    save_dict_to_json({"rows": rows}, os.path.join(save_dir, f"{split_name}_cross_modal_retrieval.json"))
    if not df.empty:
        rcol = f"recall@{ks[0]}"
        plt.close("all")
        plt.figure(figsize=(8, 4.2))
        plt.bar(df["direction"].astype(str), df[rcol].astype(float))
        plt.xticks(rotation=30, ha="right"); plt.ylabel(rcol)
        plt.title(f"{split_name} cross-modal retrieval {rcol}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_cross_modal_retrieval.png"), dpi=160)
        plt.close()
    logging.info("Saved cross-modal retrieval metrics to %s", csv_path)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# NEW: t-SNE feature distribution (Fig 8 style)
# Three panels: "Fusion All", "CXRs + Respiratory sounds", "CXRs only",
# colored by Alive / Dead. Saved to results_multimodel.
# ═══════════════════════════════════════════════════════════════════════
def _fused_under_config(model, dataloader, device, config):
    target, old = _temporarily_set_model_flags(
        model,
        use_text=config.get("text", True),
        use_tab=config.get("tabular", True),
        use_audio=config.get("audio", True))
    target.eval()
    feats, labels = [], []
    try:
        with torch.inference_mode():
            for batch in dataloader:
                batch = _apply_batch_modality_config(batch, config)
                vq = batch["visual_query"].to(device, non_blocking=True)
                im = batch["image_mask"].to(device, non_blocking=True)
                aq = batch.get("audio_query"); am = batch.get("audio_mask")
                if isinstance(aq, torch.Tensor):
                    aq = aq.to(device, non_blocking=True)
                if isinstance(am, torch.Tensor):
                    am = am.to(device, non_blocking=True)
                tf = batch.get("tab_features")
                if tf is not None:
                    tf = tf.to(device, non_blocking=True)
                with amp_autocast(device):
                    _, embeds = target(batch["textual_query"], vq, im, tf,
                                       audio_query=aq, audio_mask=am,
                                       return_embeds=True)
                feats.append(embeds["fused"].detach().cpu().float().numpy())
                labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
    finally:
        _restore_model_flags(target, old)
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    return X, np.asarray(labels, dtype=int)


def plot_tsne_feature_distributions(model, dataloader, device, save_dir, split_name,
                                    use_audio=True, seed=72):
    ensure_dir(save_dir)
    if not _HAVE_TSNE:
        logging.warning("scikit-learn TSNE unavailable; skipping t-SNE plot.")
        return None
    configs = [
        ("Fusion All", {"image": True, "text": True, "tabular": True, "audio": bool(use_audio)}),
        ("CXRs + Respiratory sounds", {"image": True, "text": False, "tabular": False, "audio": bool(use_audio)}),
        ("CXRs only", {"image": True, "text": False, "tabular": False, "audio": False}),
    ]
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    any_plotted = False
    for ax, (title, cfg) in zip(axes, configs):
        X, y = _fused_under_config(model, dataloader, device, cfg)
        if X.shape[0] < 5:
            ax.set_title(f"{title}\n(insufficient samples)")
            ax.axis("off")
            continue
        perp = float(max(5, min(30, (X.shape[0] - 1) // 3)))
        try:
            Z = TSNE(n_components=2, perplexity=perp, init="pca",
                     learning_rate="auto", random_state=seed).fit_transform(X)
        except Exception as e:
            logging.warning("t-SNE failed for %s: %s", title, e)
            ax.set_title(f"{title}\n(t-SNE failed)"); ax.axis("off"); continue
        alive = y == 1; dead = y == 0
        ax.scatter(Z[alive, 0], Z[alive, 1], s=18, alpha=0.7, label="Alive", c="#3b0f70")
        ax.scatter(Z[dead, 0], Z[dead, 1], s=18, alpha=0.7, label="Dead", c="#f0c808")
        ax.set_title(title); ax.set_xlabel("t-SNE component 1"); ax.set_ylabel("t-SNE component 2")
        ax.legend(loc="best", fontsize=8)
        any_plotted = True
        pd.DataFrame({"tsne_1": Z[:, 0], "tsne_2": Z[:, 1], "label": y}).to_csv(
            os.path.join(save_dir, f"{split_name}_tsne_{title.replace(' ', '_').replace('+', 'plus')}.csv"),
            index=False)
    fig.suptitle("Feature distribution of ICU mortality predictions using different modalities")
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{split_name}_tsne_feature_distributions.png")
    fig.savefig(out_path, dpi=160); plt.close(fig)
    if any_plotted:
        logging.info("Saved t-SNE feature distribution figure to %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# NEW: respiratory-sound spectrograms, alive vs dead (Fig 2 style)
# Panels grouped by Sex (if available) x Survival (No/Yes). Needs librosa
# and real audio files. Saved to results_multimodel.
# ═══════════════════════════════════════════════════════════════════════
def _pick_audio_example(df_all, patient_audio, label_value, sex_value=None):
    cand = df_all[df_all["90 days survival"] == label_value]
    if sex_value is not None and "Sex" in df_all.columns:
        cand = cand[cand["Sex"].astype(str).str.lower().str.startswith(str(sex_value).lower()[:1])]
    for _, row in cand.iterrows():
        pid = int(row["Patient number"])
        paths = patient_audio.get(pid, [])
        if paths:
            return pid, paths[0]
    return None, None


def plot_respiratory_spectrograms(df_all, patient_audio, save_dir,
                                  audio_sr=44100, audio_seconds=15.0,
                                  n_fft=2048, hop_length=512, n_mels=128):
    ensure_dir(save_dir)
    if not _HAVE_LIBROSA:
        logging.warning("librosa unavailable; skipping spectrogram figure.")
        return None
    try:
        import librosa.display  # noqa: F401
    except Exception as e:
        logging.warning("librosa.display unavailable: %s", e)
        return None
    has_sex = "Sex" in df_all.columns
    if has_sex:
        groups = [("Female", "No", 0), ("Male", "No", 0),
                  ("Female", "Yes", 1), ("Male", "Yes", 1)]
        nrows, ncols = 2, 2
    else:
        groups = [("All", "No", 0), ("All", "Yes", 1)]
        nrows, ncols = 1, 2
    plt.close("all")
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 6 if has_sex else 3.4))
    axes = np.atleast_1d(axes).ravel()
    plotted = 0
    for ax, (sex, surv_txt, surv_val) in zip(axes, groups):
        pid, path = _pick_audio_example(
            df_all, patient_audio, surv_val,
            sex_value=(sex if has_sex and sex != "All" else None))
        if path is None:
            ax.set_title(f"{sex}; Survival: {surv_txt}\n(no audio)"); ax.axis("off"); continue
        try:
            wav, _ = librosa.load(path, sr=audio_sr, mono=True,
                                  duration=audio_seconds)
            S = librosa.feature.melspectrogram(
                y=wav, sr=audio_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(
                S_db, sr=audio_sr, hop_length=hop_length,
                x_axis="time", y_axis="mel", ax=ax, cmap="hot")
            ax.set_ylim(0, 2000)
            ax.set_title(f"{sex}; Survival: {surv_txt}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)")
            plotted += 1
        except Exception as e:
            logging.warning("Spectrogram failed for patient %s: %s", pid, e)
            ax.set_title(f"{sex}; Survival: {surv_txt}\n(failed)"); ax.axis("off")
    fig.suptitle("Spectrograms of respiratory sounds for alive and dead samples")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "respiratory_spectrograms_alive_vs_dead.png")
    fig.savefig(out_path, dpi=160); plt.close(fig)
    if plotted:
        logging.info("Saved respiratory spectrogram figure to %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# NEW (W5): Unimodal + feature-concatenation baselines
# Extracts frozen per-modality embeddings from the trained model, then fits
# lightweight Logistic Regression heads. This gives honest, cheap baselines
# (image only / clinical only / audio only / concat) without retraining the
# heavy encoders. Saved to results_multimodel.
# ═══════════════════════════════════════════════════════════════════════
def _logreg_eval(X_tr, y_tr, X_te, y_te):
    if X_tr.shape[0] < 4 or len(np.unique(y_tr)) < 2:
        return {"auc": 0.0, "ap": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "accuracy": 0.0}
    Xtr = np.nan_to_num(X_tr); Xte = np.nan_to_num(X_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), y_tr)
    if len(np.unique(y_te)) < 2:
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        return {"auc": 0.0, "ap": 0.0,
                "f1": float(f1_score(y_te, (p >= 0.5).astype(int), zero_division=0)),
                "balanced_accuracy": 0.0, "accuracy": float(accuracy_score(y_te, (p >= 0.5).astype(int)))}
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    yhat = (p >= 0.5).astype(int)
    return {"auc": safe_binary_auc(y_te, p), "ap": safe_binary_ap(y_te, p),
            "f1": float(f1_score(y_te, yhat, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, yhat)),
            "accuracy": float(accuracy_score(y_te, yhat))}


def run_unimodal_and_concat_baselines(model, train_loader, val_loader,
                                      device, save_dir, split_name):
    """
    Baselines required by W5:
      - image only, clinical only (text+tabular), audio only
      - feature-concatenation multimodal (image+text+tabular+audio)
    All use frozen embeddings from the trained CoCross encoders + LogReg.
    """
    ensure_dir(save_dir)
    emb_tr, _, y_tr, _ = collect_embeddings(model, train_loader, device)
    emb_te, _, y_te, _ = collect_embeddings(model, val_loader, device)

    def cat(parts_dict, keys):
        arrs = [parts_dict[k] for k in keys if k in parts_dict and parts_dict[k].size]
        if not arrs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(arrs, axis=1)

    baseline_specs = {
        "image_only": ["img"],
        "clinical_only_text_tabular": ["txt", "tab"],
        "audio_only": ["aud"],
        "concat_all": ["img", "txt", "tab", "aud"],
    }
    rows = []
    for name, keys in baseline_specs.items():
        Xtr = cat(emb_tr, keys); Xte = cat(emb_te, keys)
        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
            continue
        m = _logreg_eval(Xtr, y_tr, Xte, y_te)
        rows.append({"baseline": name, "modalities": "+".join(keys), **m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{split_name}_baselines.csv"), index=False)
    save_dict_to_json({"rows": rows}, os.path.join(save_dir, f"{split_name}_baselines.json"))
    if not df.empty:
        plt.close("all")
        plt.figure(figsize=(8, 4.2))
        plt.bar(df["baseline"].astype(str), df["auc"].astype(float))
        plt.xticks(rotation=25, ha="right"); plt.ylabel("AUC")
        plt.title(f"{split_name} baseline comparison (frozen-feature LogReg)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_baselines.png"), dpi=160)
        plt.close()
    logging.info("Saved baseline comparison to %s", save_dir)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# NEW: tabular permutation importance (clinical interpretability)
# Per-feature drop in AUC when a tabular feature is shuffled. Saved to
# results_multimodel. Extends interpretability to the clinical modality.
# ═══════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def _model_probs_with_tab(model, batches, device, perm_idx=None, perm_order=None):
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.eval()
    probs, labels = [], []
    for batch in batches:
        vq = batch["visual_query"].to(device)
        im = batch["image_mask"].to(device)
        aq = batch.get("audio_query"); am = batch.get("audio_mask")
        if isinstance(aq, torch.Tensor):
            aq = aq.to(device)
        if isinstance(am, torch.Tensor):
            am = am.to(device)
        tf = batch.get("tab_features")
        if tf is not None:
            tf = tf.clone().to(device)
            if perm_idx is not None and perm_order is not None and tf.shape[0] == len(perm_order):
                tf[:, perm_idx] = tf[perm_order, perm_idx]
        with amp_autocast(device):
            z = target(batch["textual_query"], vq, im, tf,
                       audio_query=aq, audio_mask=am, return_embeds=False)
        probs.extend(torch.sigmoid(sanitize_tensor(z)).cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().astype(int).tolist())
    return np.asarray(probs), np.asarray(labels, dtype=int)


def run_tabular_permutation_importance(model, val_loader, device, tab_cols,
                                       save_dir, split_name, n_repeats=5, seed=72):
    ensure_dir(save_dir)
    batches = [dict(b) for b in val_loader]
    if not batches:
        return []
    # cache tab to per-batch so permutation is within-batch
    base_p, y = _model_probs_with_tab(model, batches, device)
    base_auc = safe_binary_auc(y, base_p)
    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(tab_cols):
        drops = []
        for _ in range(n_repeats):
            per_batch_probs, per_batch_y = [], []
            for b in batches:
                tf = b.get("tab_features")
                n = tf.shape[0] if isinstance(tf, torch.Tensor) else 0
                order = rng.permutation(n) if n > 0 else None
                p, yy = _model_probs_with_tab(model, [b], device, perm_idx=j, perm_order=order)
                per_batch_probs.extend(p.tolist()); per_batch_y.extend(yy.tolist())
            auc_perm = safe_binary_auc(np.asarray(per_batch_y), np.asarray(per_batch_probs))
            drops.append(base_auc - auc_perm)
        rows.append({"feature": col, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_std": float(np.std(drops)), "base_auc": float(base_auc)})
    df = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)
    df.to_csv(os.path.join(save_dir, f"{split_name}_tabular_permutation_importance.csv"), index=False)
    if not df.empty:
        plt.close("all")
        plt.figure(figsize=(8, 4.2))
        plt.barh(df["feature"].astype(str), df["auc_drop_mean"].astype(float),
                 xerr=df["auc_drop_std"].astype(float))
        plt.gca().invert_yaxis(); plt.xlabel("AUC drop when permuted")
        plt.title(f"{split_name} clinical feature importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_tabular_permutation_importance.png"), dpi=160)
        plt.close()
    logging.info("Saved tabular permutation importance to %s", save_dir)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# NEW: audio input-gradient saliency (audio interpretability)
# Saliency over the mel-spectrogram of the audio encoder front-end. Saved to
# results_multimodel.
# ═══════════════════════════════════════════════════════════════════════
def save_audio_saliency(model, dataloader, device, save_dir, split_name, max_samples=6):
    ensure_dir(save_dir)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    if not getattr(target, "base_use_audio", False) or target.audio_encoder is None:
        logging.info("Audio disabled; skipping audio saliency.")
        return []
    target.eval()
    saved, rows = 0, []
    for batch in dataloader:
        if saved >= int(max_samples):
            break
        aq = batch.get("audio_query")
        if not isinstance(aq, torch.Tensor) or aq.numel() == 0:
            continue
        aq = aq.to(device).clone().detach().requires_grad_(True)
        am = batch.get("audio_mask")
        if isinstance(am, torch.Tensor):
            am = am.to(device)
        vq = batch["visual_query"].to(device)
        im = batch["image_mask"].to(device)
        tf = batch.get("tab_features")
        if tf is not None:
            tf = tf.to(device)
        target.zero_grad(set_to_none=True)
        with safe_no_amp_context(device):
            logits = target(batch["textual_query"], vq, im, tf,
                            audio_query=aq, audio_mask=am, return_embeds=False)
            score = logits.sum()
        try:
            score.backward()
        except Exception as e:
            logging.warning("Audio saliency backward failed: %s", e)
            break
        if aq.grad is None:
            break
        sal = aq.grad.detach().abs()  # [B, K, T]
        pids = batch.get("patient_id")
        if isinstance(pids, torch.Tensor):
            pids = pids.cpu().numpy().astype(int).tolist()
        else:
            pids = list(range(aq.shape[0]))
        for bi in range(aq.shape[0]):
            if saved >= int(max_samples):
                break
            if isinstance(am, torch.Tensor) and am.numel():
                valid = torch.where(am[bi].detach().cpu() > 0.5)[0]
                if len(valid) == 0:
                    continue
                ki = int(valid[0].item())
            else:
                ki = 0
            s = sal[bi, ki].cpu().numpy()
            plt.close("all")
            plt.figure(figsize=(9, 3.2))
            plt.plot(s, linewidth=0.6)
            plt.xlabel("Sample index"); plt.ylabel("|grad|")
            plt.title(f"Audio saliency (patient {int(pids[bi])})")
            plt.tight_layout()
            path = os.path.join(save_dir, f"{split_name}_patient_{int(pids[bi])}_audio_saliency.png")
            plt.savefig(path, dpi=140); plt.close()
            rows.append({"patient_id": int(pids[bi]), "path": path})
            saved += 1
    target.zero_grad(set_to_none=True)
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, f"{split_name}_audio_saliency_index.csv"), index=False)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Model construction / dataset helpers for cross-validation
# ═══════════════════════════════════════════════════════════════════════
def _build_model(shared, cfg, tab_dim, device, clip_weight):
    """Build a fresh BiomedAudioCLIP for one fold. The CLIP image tower is
    deep-copied so per-fold fine-tuning never leaks across folds. A fresh
    OfflineTextEncoder is created and its vocab is fit on TRAIN texts only."""
    clip_model = copy.deepcopy(shared["clip_model"])
    offline_text = None
    if shared["offline_text_template"] is not None:
        offline_text = OfflineTextEncoder(embed_dim=512)
        if shared.get("train_texts"):
            offline_text.fit_vocab(shared["train_texts"])
    model = BiomedAudioCLIP(
        clip_model=clip_model,
        clip_tokenizer=shared["clip_tokenizer"],
        offline_text_encoder=offline_text,
        embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        freeze_biomedclip=cfg["freeze_biomedclip"],
        use_text=cfg["use_text"], use_tab=cfg["use_tab"],
        use_audio=cfg["use_audio"], tab_dim=tab_dim,
        use_cross_attn=cfg["use_cross_attn"],
        audio_sr=cfg["audio_sr"], audio_n_fft=cfg["audio_n_fft"],
        audio_hop=cfg["audio_hop"], audio_n_mels=cfg["audio_n_mels"],
        use_pretrained_esresnext=cfg["use_pretrained_esresnext"],
        esresnext_repo_dir=cfg["esresnext_repo_dir"],
        esresnext_weights_path=cfg["esresnext_weights_path"],
        freeze_esresnext_backbone=cfg["freeze_esresnext_backbone"],
        init_clip_temp=cfg["init_clip_temp"]).to(device)
    return model


def _make_ds_common(base_path, df, transform, cfg, patient_images,
                    patient_audio, tab_cols, is_train):
    return CoCrossDataset(
        base_path=base_path, metadata_df=df, transform=transform,
        fixed_images=cfg["fixed_images"],
        borrow_images=cfg["borrow_images"],
        image_subdirs=cfg["image_subdirs"],
        patient_images=patient_images,
        use_audio=cfg["use_audio"], audio_subdir=cfg["audio_subdir"],
        fixed_audio=cfg["fixed_audio"], audio_sr=cfg["audio_sr"],
        audio_seconds=cfg["audio_seconds"], is_train=is_train,
        patient_audio=patient_audio, tab_cols=tab_cols)


def _make_optimizer(model, base_lr, weight_decay):
    decay_p, no_decay_p = [], []
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, p in target.named_parameters():
        if not p.requires_grad:
            continue
        if (name.endswith(".bias") or any(
                k in name.lower()
                for k in ("layernorm", "layer_norm", "bn", "logit_scale"))):
            no_decay_p.append(p)
        else:
            decay_p.append(p)
    return optim.AdamW(
        [{"params": decay_p, "lr": base_lr, "weight_decay": weight_decay},
         {"params": no_decay_p, "lr": base_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.98), eps=1e-6)


def _train_and_eval_fold(fold, train_idx, val_idx, df_pat, base_path,
                         shared, cfg, tab_cols, device, model_dir,
                         patient_images, patient_audio,
                         run_reports=False):
    fold_seed_offset = (fold if isinstance(fold, int)
                        else int(hashlib.sha1(str(fold).encode()).hexdigest()[:6], 16))
    set_seed(cfg["seed"] + fold_seed_offset)
    fold_dir = ensure_dir(os.path.join(model_dir, f"fold_{fold}"))
    set_logger(os.path.join(fold_dir, "train.log"))
    logging.info("===== Fold %d =====", fold)

    df_train = df_pat.iloc[train_idx].reset_index(drop=True)
    df_val = df_pat.iloc[val_idx].reset_index(drop=True)
    df_train, df_val, tab_imputer, tab_scaler = attach_tabular(tab_cols, df_train, df_val)

    shared = dict(shared)
    shared["train_texts"] = [
        build_patient_text(r, [c for c in OPTIONAL_TAB_CANDIDATES if c in df_train.columns])
        for _, r in df_train.iterrows()]

    train_tf = _XRayAugWrap(shared["val_tf"]) if cfg["augment"] else shared["val_tf"]
    train_ds = _make_ds_common(base_path, df_train, train_tf, cfg,
                               patient_images, patient_audio, tab_cols, True)
    val_ds = _make_ds_common(base_path, df_val, shared["val_tf"], cfg,
                             patient_images, patient_audio, tab_cols, False)

    labels_train = df_train["90 days survival"].values.astype(int)
    sampler = None
    if cfg["balance_strategy"] in ("sampler", "both"):
        w = get_sample_weights(labels_train, strategy="effective")
        sampler = WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(w),
            replacement=True)

    # drop_last avoids a singleton final batch reaching any trainable
    # BatchNorm (e.g. a from-scratch or unfrozen audio encoder). Guarded so a
    # fold smaller than one batch is never emptied.
    train_drop_last = len(train_ds) > cfg["batch_size"]
    train_loader = make_loader(
        train_ds, cfg["batch_size"],
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg["num_workers"], drop_last=train_drop_last)
    val_loader = make_loader(val_ds, cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    pos_weight = None
    if cfg["balance_strategy"] in ("loss", "both"):
        n_pos = max(1, int(labels_train.sum()))
        n_neg = max(1, int((1 - labels_train).sum()))
        pos_weight = n_neg / n_pos
    criterion = CombinedLoss(pos_weight=pos_weight).to(device)

    model = _build_model(shared, cfg, len(tab_cols), device, cfg["clip_weight"])
    set_classifier_prior_bias(model, float(np.mean(labels_train)))
    optimizer = _make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.05)
    scaler = AmpGradScaler(enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=0.999, device=device) if cfg["use_ema"] else None

    best_score, best_state, history = -1e9, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        if cfg["progressive_unfreeze"]:
            n_unf = model.progressive_unfreeze(
                epoch, unfreeze_at=cfg["unfreeze_at"], enabled=True)
            if n_unf > 0:
                optimizer = reset_optimizer_after_unfreeze(
                    model, cfg["lr"] * 0.1, cfg["weight_decay"])
        tr_loss = train_one_epoch(
            model, optimizer, train_loader, scaler, device, criterion,
            label_smoothing=cfg["label_smoothing"],
            use_label_smoothing=cfg["use_label_smoothing"],
            log_every=0, ema=ema, clip_weight=cfg["clip_weight"],
            accumulation_steps=cfg["accumulation_steps"])
        eval_model = ema.ema if ema is not None else model
        metrics, p, y, lg, ids, _ = evaluate(
            eval_model, val_loader, device, criterion, threshold=0.5)
        scheduler.step()
        score = validation_selection_score(metrics, probs=p)
        history.append({
            "epoch": epoch, "train_loss": float(tr_loss),
            "val_loss": metrics["loss"], "val_auc": metrics["auc"],
            "val_f1": metrics["f1"], "val_recall": metrics["recall"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "selection_score": score})
        logging.info("Fold %d epoch %d | train %.4f | val AUC %.3f "
                     "F1 %.3f BAcc %.3f | score %.4f",
                     fold, epoch, tr_loss, metrics["auc"], metrics["f1"],
                     metrics["balanced_accuracy"], score)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(
                (ema.ema if ema is not None else model).state_dict())

    if best_state is not None:
        (ema.ema if ema is not None else model).load_state_dict(best_state)
    final_model = ema.ema if ema is not None else model

    # Calibrate on validation, choose threshold
    _, p_val, y_val, lg_val, ids_val, _ = evaluate(
        final_model, val_loader, device, criterion, threshold=0.5)
    temp = temperature_scaling(lg_val, y_val)
    p_cal = apply_temperature_to_logits(lg_val, temp)
    best_t, _ = find_best_threshold(y_val, p_cal, optimize_for="balanced_accuracy")

    plot_training_history(history, fold_dir, f"fold{fold}")
    metrics_final = save_standard_eval_artifacts(
        y_val, p_cal, ids_val, best_t, fold_dir, f"fold{fold}_val")
    metrics_final.update({"fold": fold, "temperature": temp, "threshold": best_t})
    save_metrics_json(metrics_final, os.path.join(fold_dir, f"fold{fold}_summary.json"))

    if run_reports:
        rep_dir = ensure_dir(os.path.join(fold_dir, "reports"))
        try:
            run_modality_ablation_suite(
                final_model, val_loader, device, criterion, temp, None,
                best_t, rep_dir, f"fold{fold}", use_audio=cfg["use_audio"])
        except Exception as e:
            logging.warning("Ablation suite failed: %s", e)
        try:
            collect_and_save_attention_weights(
                final_model, val_loader, device, rep_dir, f"fold{fold}")
        except Exception as e:
            logging.warning("Attention logging failed: %s", e)
        try:
            save_modality_shap_values(
                final_model, val_loader, device, rep_dir, f"fold{fold}",
                max_samples=cfg["shap_max_samples"])
        except Exception as e:
            logging.warning("Modality SHAP failed: %s", e)
        try:
            gradcam_loader = val_loader
            gradcam_ids = [int(x) for x in cfg.get("gradcam_patient_ids", [])]
            if gradcam_ids:
                df_grad = df_pat[df_pat["Patient number"].astype(int).isin(gradcam_ids)].copy()
                if not df_grad.empty:
                    df_grad = apply_tabular_transform(tab_cols, df_grad, tab_imputer, tab_scaler)
                    gradcam_ds = _make_ds_common(
                        base_path, df_grad.reset_index(drop=True), shared["val_tf"], cfg,
                        patient_images, patient_audio, tab_cols, False)
                    gradcam_loader = make_loader(
                        gradcam_ds, cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"])
                else:
                    logging.warning("Requested Grad-CAM patients not found in metadata: %s", gradcam_ids)
            save_gradcam_heatmaps(
                final_model, gradcam_loader, device, rep_dir, f"fold{fold}",
                max_samples=cfg["gradcam_max_samples"],
                target_patient_ids=gradcam_ids,
                make_composite=True)
        except Exception as e:
            logging.warning("Grad-CAM failed: %s", e)
        try:
            run_cross_modal_retrieval(
                final_model, val_loader, device, rep_dir, f"fold{fold}")
        except Exception as e:
            logging.warning("Retrieval eval failed: %s", e)
        try:
            plot_tsne_feature_distributions(
                final_model, val_loader, device, rep_dir, f"fold{fold}",
                use_audio=cfg["use_audio"])
        except Exception as e:
            logging.warning("t-SNE failed: %s", e)
        try:
            run_unimodal_and_concat_baselines(
                final_model, train_loader, val_loader, device, rep_dir, f"fold{fold}")
        except Exception as e:
            logging.warning("Baselines failed: %s", e)
        try:
            run_tabular_permutation_importance(
                final_model, val_loader, device, tab_cols, rep_dir, f"fold{fold}")
        except Exception as e:
            logging.warning("Tabular importance failed: %s", e)
        try:
            save_audio_saliency(
                final_model, val_loader, device, rep_dir, f"fold{fold}")
        except Exception as e:
            logging.warning("Audio saliency failed: %s", e)

    del model, final_model
    if ema is not None:
        del ema
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_final


# ═══════════════════════════════════════════════════════════════════════
# NEW (W4): CLIP / contrastive loss-weight ablation
# Trains a light single-split model for each weight in {0.0, 0.005, 0.01,
# 0.05, 0.1} and reports val AUC/AP/F1 so the chosen weight is justified.
# This is the most expensive optional module (one training run per weight),
# so it is gated behind cfg["run_clip_weight_ablation"].
# ═══════════════════════════════════════════════════════════════════════
def _format_float_for_table(x: Any, nd: int = 3) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "--"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "--"


def _write_lambda_sensitivity_tables(df: pd.DataFrame, out_dir: str):
    """Write CSV/Markdown/LaTeX versions of the CLIP lambda sensitivity table."""
    ensure_dir(out_dir)
    if df.empty:
        return
    table_df = df.copy()
    table_df = table_df.sort_values("lambda_clip").reset_index(drop=True)

    md_lines = [
        "| $\\lambda_{CLIP}$ | AUC | AP | F1 | Balanced Acc. | Recall | Best |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, r in table_df.iterrows():
        md_lines.append(
            f"| {float(r['lambda_clip']):.4g} | "
            f"{_format_float_for_table(r.get('auc'))} | "
            f"{_format_float_for_table(r.get('ap'))} | "
            f"{_format_float_for_table(r.get('f1'))} | "
            f"{_format_float_for_table(r.get('balanced_accuracy'))} | "
            f"{_format_float_for_table(r.get('recall'))} | "
            f"{'*' if bool(r.get('is_best', False)) else ''} |")
    with open(os.path.join(out_dir, "lambda_sensitivity_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity to the CLIP-style contrastive loss weight $\lambda_{CLIP}$.}",
        r"\label{tab:lambda_sensitivity}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$\lambda_{CLIP}$ & AUC & AP & F1 & BAcc & Recall \\",
        r"\midrule",
    ]
    for _, r in table_df.iterrows():
        star = r"$^{\star}$" if bool(r.get("is_best", False)) else ""
        tex.append(
            f"{float(r['lambda_clip']):.4g}{star} & "
            f"{_format_float_for_table(r.get('auc'))} & "
            f"{_format_float_for_table(r.get('ap'))} & "
            f"{_format_float_for_table(r.get('f1'))} & "
            f"{_format_float_for_table(r.get('balanced_accuracy'))} & "
            f"{_format_float_for_table(r.get('recall'))} \\\\")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(out_dir, "lambda_sensitivity_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# NEW (W4): CLIP / contrastive loss-weight ablation
# Trains a light single-split model for each lambda and reports val AUC/AP/F1
# so the chosen weight is justified. This is gated behind
# cfg["run_clip_weight_ablation"] / --run_clip_weight_ablation.
# ═══════════════════════════════════════════════════════════════════════
def run_clip_weight_ablation(df_pat, base_path, shared, cfg, tab_cols,
                             device, model_dir, patient_images, patient_audio,
                             weights=None):
    out_dir = ensure_dir(os.path.join(model_dir, "clip_weight_ablation"))
    set_logger(os.path.join(out_dir, "clip_weight_ablation.log"))
    if weights is None:
        weights = cfg.get("clip_ablation_weights",
                          [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    weights = [float(w) for w in weights]
    logging.info("CLIP lambda sensitivity ablation over weights=%s", weights)

    labels = df_pat["90 days survival"].values.astype(int)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg["clip_ablation_val_frac"], random_state=cfg["seed"])
    tr_idx, va_idx = next(sss.split(np.zeros(len(labels)), labels))

    rows = []
    for w in weights:
        sub_cfg = dict(cfg)
        sub_cfg["clip_weight"] = float(w)
        sub_cfg["epochs"] = int(cfg["clip_ablation_epochs"])
        sub_cfg["report_folds"] = []
        try:
            m = _train_and_eval_fold(
                fold=f"clipw_{w:g}", train_idx=tr_idx, val_idx=va_idx,
                df_pat=df_pat, base_path=base_path, shared=shared,
                cfg=sub_cfg, tab_cols=tab_cols, device=device,
                model_dir=out_dir, patient_images=patient_images,
                patient_audio=patient_audio, run_reports=False)
            rows.append({
                "lambda_clip": float(w),
                "clip_weight": float(w),
                "auc": float(m.get("auc", 0.0)),
                "ap": float(m.get("ap", 0.0)),
                "f1": float(m.get("f1", 0.0)),
                "balanced_accuracy": float(m.get("balanced_accuracy", 0.0)),
                "accuracy": float(m.get("accuracy", 0.0)),
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "threshold": float(m.get("threshold", 0.5)),
                "temperature": float(m.get("temperature", 1.0)),
                "status": "ok",
            })
            logging.info("lambda=%.5g | AUC %.4f AP %.4f F1 %.4f BAcc %.4f",
                         w, rows[-1]["auc"], rows[-1]["ap"],
                         rows[-1]["f1"], rows[-1]["balanced_accuracy"])
        except Exception as e:
            logging.exception("CLIP lambda ablation failed for lambda=%s", w)
            rows.append({
                "lambda_clip": float(w), "clip_weight": float(w),
                "auc": 0.0, "ap": 0.0, "f1": 0.0,
                "balanced_accuracy": 0.0, "accuracy": 0.0,
                "precision": 0.0, "recall": 0.0,
                "threshold": 0.5, "temperature": 1.0,
                "status": f"failed: {e}",
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if not df.empty:
        # Selection priority: AUC, then AP, then F1.
        best_idx = df[["auc", "ap", "f1"]].astype(float).sort_values(
            ["auc", "ap", "f1"], ascending=False).index[0]
        df["is_best"] = False
        df.loc[best_idx, "is_best"] = True
    df.to_csv(os.path.join(out_dir, "clip_weight_ablation.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "lambda_sensitivity.csv"), index=False)
    save_dict_to_json({"rows": rows}, os.path.join(out_dir, "clip_weight_ablation.json"))
    _write_lambda_sensitivity_tables(df, out_dir)

    try:
        plt.close("all")
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(df["lambda_clip"].astype(float), df["auc"].astype(float),
                 marker="o", label="AUC")
        plt.plot(df["lambda_clip"].astype(float), df["ap"].astype(float),
                 marker="s", label="AP")
        plt.plot(df["lambda_clip"].astype(float), df["f1"].astype(float),
                 marker="^", label="F1")
        plt.xscale("symlog", linthresh=0.001)
        plt.xlabel(r"CLIP loss weight $\lambda_{CLIP}$")
        plt.ylabel("Validation metric")
        plt.title(r"Sensitivity to $\lambda_{CLIP}$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "clip_weight_ablation.png"), dpi=180)
        plt.close()
    except Exception as e:
        logging.warning("Could not plot CLIP lambda ablation: %s", e)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Cross-validation driver
# ═══════════════════════════════════════════════════════════════════════
def run_cross_validation(df_pat, base_path, shared, cfg, tab_cols, device,
                         model_dir, patient_images, patient_audio):
    labels = df_pat["90 days survival"].values.astype(int)
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True,
                          random_state=cfg["seed"])
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        run_reports = (fold in cfg["report_folds"])
        m = _train_and_eval_fold(
            fold, tr, va, df_pat, base_path, shared, cfg, tab_cols,
            device, model_dir, patient_images, patient_audio,
            run_reports=run_reports)
        fold_metrics.append(m)

    if fold_metrics:
        keys = ["auc", "ap", "f1", "balanced_accuracy", "accuracy",
                "precision", "recall", "brier", "ece"]
        summary = {}
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in fold_metrics]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
        save_dict_to_json(
            {"per_fold": fold_metrics, "summary": summary},
            os.path.join(model_dir, "cv_summary.json"))
        pd.DataFrame(fold_metrics).to_csv(
            os.path.join(model_dir, "cv_per_fold_metrics.csv"), index=False)
        logging.info("CV done | AUC %.3f +/- %.3f | AP %.3f +/- %.3f | "
                     "F1 %.3f +/- %.3f",
                     summary["auc_mean"], summary["auc_std"],
                     summary["ap_mean"], summary["ap_std"],
                     summary["f1_mean"], summary["f1_std"])
    return fold_metrics


# ═══════════════════════════════════════════════════════════════════════
# Default config + notes file
# ═══════════════════════════════════════════════════════════════════════
def default_config() -> Dict[str, Any]:
    return {
        # paths (overridden by CLI args in main())
        "base_path": "CoCross",
        "metadata_path": "CoCross/Metadata.xlsx",
        "audio_root": None,
        "biomedclip_dir": "./BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "model_dir": "./results_multimodel",
        "allow_tf32": True,
        # data
        "image_subdirs": ["XR"], "audio_subdir": "AUDIO",
        "fixed_images": 4, "fixed_audio": 1,
        "borrow_images": "none",
        "audio_sr": 44100, "audio_seconds": 10.0,
        "audio_n_fft": 2048, "audio_hop": 512, "audio_n_mels": 128,
        # model
        "embed_dim": 512, "hidden_dim": 512, "dropout": 0.30,
        "freeze_biomedclip": True, "use_text": True, "use_tab": True,
        "use_audio": True, "use_cross_attn": True,
        "use_pretrained_esresnext": True,
        "esresnext_repo_dir": "./ESResNeXt-fbsp",
        "esresnext_weights_path": "./ESResNeXt-fbsp/ESResNeXtFBSP_AudioSet.pt",
        "freeze_esresnext_backbone": True, "init_clip_temp": 0.07,
        # training
        "seed": 72, "n_splits": 5, "epochs": 40, "batch_size": 8,
        "lr": 2e-4, "weight_decay": 1e-2, "label_smoothing": 0.07,
        "use_label_smoothing": True, "augment": True, "use_ema": True,
        "accumulation_steps": 1, "num_workers": 0,
        "clip_weight": 0.005,
        "progressive_unfreeze": False, "unfreeze_at": 30,
        # FIX: single balancing strategy. Choose 'sampler' (default), 'loss',
        # 'both' (old buggy behavior), or 'none'.
        "balance_strategy": "sampler",
        # reporting
        "report_folds": [1],
        "shap_max_samples": 8, "gradcam_max_samples": 8,
        "gradcam_patient_ids": [16, 20, 22, 24],
        # optional heavy experiments
        "run_clip_weight_ablation": False,
        "clip_ablation_epochs": 15, "clip_ablation_val_frac": 0.25,
        "clip_ablation_weights": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        "run_spectrograms": True,
    }


def write_notes_file(model_dir, cfg, df_pat, tab_cols):
    ensure_dir(model_dir)
    lines = [
        "CoCross / BiomedAudioCLIP results notes",
        "=" * 44, "",
        "Patients: %d | tabular columns: %s" % (len(df_pat), ", ".join(tab_cols)),
        "Output directory: %s" % os.path.abspath(model_dir), "",
        "FIXES APPLIED IN THIS VERSION",
        "-" * 30,
        "1. Vocab mismatch: OfflineTextEncoder vocabulary is now fit on the",
        "   exact text format produced by the dataset (build_patient_text),",
        "   including optional lab columns. Optional field names no longer",
        "   collapse to [UNK]. Vocab is fit per fold on TRAIN texts only.",
        "2. Double class-balancing removed: balance_strategy controls whether",
        "   the WeightedRandomSampler OR loss pos_weight is used. Default is",
        "   'sampler' only. 'both' reproduces the old behavior.",
        "3. DICOM windowing: chest X-rays now use VOI WindowCenter/WindowWidth",
        "   tags with a 1-99 percentile min-max fallback (CT Hounsfield windows",
        "   removed). MONOCHROME1 is inverted. Single plane replicated to RGB.",
        "4. Audio embedding is renormalized after attention pooling so the",
        "   contrastive (CLIP) loss compares unit-norm vectors across modalities.",
        "5. Text encoder depth reduced (6 -> 2 layers) to limit overfitting on",
        "   ~110 training patients.",
        "6. find_best_threshold no longer recomputes AUC inside the loop.", "",
        "KNOWN CAVEATS (report honestly in the paper)",
        "-" * 30,
        "A. Text/tabular redundancy: the clinical 'text' is a serialization of",
        "   the same tabular variables, so 'no_text' and 'no_tabular' ablation",
        "   rows share information. Treat modality-importance conclusions with",
        "   that caveat, or replace text with genuine free-text notes.",
        "B. W3 (audio): ESResNeXt-fbsp is AudioSet-pretrained, not lung-sound",
        "   adapted. For the camera-ready, run intermediate fine-tuning on ICBHI",
        "   2017 and point esresnext_weights_path at the adapted checkpoint; the",
        "   loader transfers any compatible tensors automatically.",
        "C. Small-dataset framing (W1): report AUC/AP (threshold-free) as primary",
        "   metrics; treat F1/BAcc/recall as secondary with mean +/- std across",
        "   folds. Consider repeated stratified CV for tighter estimates.", "",
        "ARTIFACTS GENERATED",
        "-" * 30,
        "- modality_coverage_table.csv/.png/.json           (W1)",
        "- fold_*/reports/*_cross_modal_retrieval.csv/.png   (W2: Recall@k, MRR)",
        "- clip_weight_ablation/clip_weight_ablation.csv     (W4, if enabled)",
        "- clip_weight_ablation/lambda_sensitivity.csv       (lambda sensitivity table)",
        "- clip_weight_ablation/lambda_sensitivity_table.tex (paper-ready lambda table)",
        "- fold_*/reports/*_baselines.csv/.png               (W5)",
        "- fold_*/reports/*_modality_ablation_7configs.csv   (ablation study)",
        "- respiratory_spectrograms_alive_vs_dead.png        (spectrograms)",
        "- fold_*/reports/*_tsne_feature_distributions.png   (t-SNE Fig 8)",
        "- fold_*/reports/*_gradcam.png                      (individual Grad-CAMs)",
        "- fold_*/reports/*_gradcam_composite_2x4.png       (patients 16/20/22/24 composite)",
        "- fold_*/reports/*_modality_shap_*                  (SHAP, all modalities)",
        "- fold_*/reports/*_cross_modal_attention_*          (attention logging)",
        "- fold_*/reports/*_tabular_permutation_importance.* (clinical XAI)",
        "- fold_*/reports/*_audio_saliency_*                 (audio XAI)",
        "- cv_summary.json, cv_per_fold_metrics.csv          (main results)",
    ]
    with open(os.path.join(model_dir, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
def build_shared_resources(cfg, device):
    """Load BiomedCLIP once and share its (deep-copied per fold) image tower."""
    (clip_model, train_tf, val_tf, clip_tokenizer,
     offline_text_encoder, has_text) = load_pretrained_biomedclip(
        model_name=cfg["biomedclip_dir"], cache_dir=None)
    clip_model = clip_model.to(device)
    return {
        "clip_model": clip_model, "clip_tokenizer": clip_tokenizer,
        "offline_text_template": offline_text_encoder,
        "has_text": has_text, "train_tf": train_tf, "val_tf": val_tf,
        "train_texts": None,
    }


def run_pipeline(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = ensure_dir(cfg["model_dir"])
    set_logger(os.path.join(model_dir, "pipeline.log"))
    _maybe_enable_tf32(cfg.get("allow_tf32", True))
    set_seed(cfg["seed"])
    logging.info("Device: %s | output dir: %s", device, os.path.abspath(model_dir))

    # 1. metadata + patient-level collapse
    df_raw, tab_cols = load_full_metadata(cfg["metadata_path"])
    df_pat = collapse_to_patient_level(
        df_raw, tab_cols, agg="mean", label_mode="majority")
    logging.info("Loaded %d patients (after patient-level collapse).", len(df_pat))

    # 2. scan modalities
    patient_images = scan_all_patient_images(
        cfg["base_path"], df_pat, cfg["image_subdirs"])
    patient_audio = (scan_all_patient_audio(
        cfg["base_path"], df_pat, audio_subdir=cfg["audio_subdir"],
        audio_root=cfg.get("audio_root"))
        if cfg["use_audio"] else {})
    log_modality_coverage(df_pat, patient_images, patient_audio,
                          use_audio=cfg["use_audio"])

    # 3. W1 coverage table
    save_modality_coverage_table(
        df_pat, patient_images, patient_audio, model_dir,
        use_audio=cfg["use_audio"])

    # 4. spectrograms (dataset-level, once)
    if cfg.get("run_spectrograms", True):
        try:
            plot_respiratory_spectrograms(
                df_pat, patient_audio, model_dir,
                audio_sr=cfg["audio_sr"], audio_seconds=cfg["audio_seconds"],
                n_fft=cfg["audio_n_fft"], hop_length=cfg["audio_hop"],
                n_mels=cfg["audio_n_mels"])
        except Exception as e:
            logging.warning("Spectrogram generation failed: %s", e)

    # 5. shared encoders
    shared = build_shared_resources(cfg, device)

    # 6. cross-validation (+ per-fold reports incl. retrieval, t-SNE, baselines)
    run_cross_validation(
        df_pat, cfg["base_path"], shared, cfg, tab_cols, device,
        model_dir, patient_images, patient_audio)

    # 7. W4 CLIP loss-weight ablation (optional, expensive)
    if cfg.get("run_clip_weight_ablation", False):
        try:
            run_clip_weight_ablation(
                df_pat, cfg["base_path"], shared, cfg, tab_cols, device,
                model_dir, patient_images, patient_audio)
        except Exception as e:
            logging.warning("CLIP weight ablation failed: %s", e)

    # 8. notes
    write_notes_file(model_dir, cfg, df_pat, tab_cols)
    logging.info("Pipeline complete. All artifacts in %s", os.path.abspath(model_dir))


def main():
    import argparse
    cfg = default_config()
    p = argparse.ArgumentParser(
        description="BiomedAudioCLIP / CoCross tri-modal ICU survival pipeline")
    p.add_argument("--base_path", default="CoCross",
                   help="Root folder containing per-patient subfolders.")
    p.add_argument("--metadata_path", default="CoCross/Metadata.xlsx")
    p.add_argument("--audio_root", default=None,
                   help="Optional global audio root if not under base_path.")
    p.add_argument("--biomedclip_dir",
                   default="./BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--model_dir", default="./results_multimodel",
                   help="All outputs are written here (relative dir).")
    p.add_argument("--epochs", type=int, default=cfg["epochs"])
    p.add_argument("--n_splits", type=int, default=cfg["n_splits"])
    p.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    p.add_argument("--lr", type=float, default=cfg["lr"])
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--num_workers", type=int, default=cfg["num_workers"])
    p.add_argument("--clip_weight", type=float, default=cfg["clip_weight"])
    p.add_argument("--balance_strategy", default=cfg["balance_strategy"],
                   choices=["sampler", "loss", "both", "none"])
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--report_folds", default="1",
                   help="Comma-separated fold numbers to run heavy reports on.")
    p.add_argument("--run_clip_weight_ablation", action="store_true",
                   help="Run the W4 contrastive-weight ablation (slow).")
    p.add_argument("--clip_ablation_weights", default=None,
                   help="Comma-separated lambda values for --run_clip_weight_ablation, e.g. 0,0.001,0.005,0.01,0.02,0.05,0.1.")
    p.add_argument("--clip_ablation_epochs", type=int, default=cfg["clip_ablation_epochs"],
                   help="Epochs per lambda value in the CLIP-weight ablation.")
    p.add_argument("--gradcam_patient_ids", default="16,20,22,24",
                   help="Comma-separated patient IDs for the 2x4 Grad-CAM composite.")
    p.add_argument("--no_spectrograms", action="store_true")
    args = p.parse_args()

    cfg.update({
        "base_path": args.base_path,
        "metadata_path": args.metadata_path,
        "audio_root": args.audio_root,
        "biomedclip_dir": args.biomedclip_dir,
        "model_dir": args.model_dir,
        "epochs": args.epochs, "n_splits": args.n_splits,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "num_workers": args.num_workers, "clip_weight": args.clip_weight,
        "balance_strategy": args.balance_strategy,
        "use_audio": not args.no_audio, "use_text": not args.no_text,
        "report_folds": [int(x) for x in str(args.report_folds).split(",") if x.strip()],
        "run_clip_weight_ablation": args.run_clip_weight_ablation,
        "clip_ablation_epochs": args.clip_ablation_epochs,
        "gradcam_patient_ids": [int(x) for x in str(args.gradcam_patient_ids).split(",") if x.strip()],
        "run_spectrograms": not args.no_spectrograms,
    })
    if args.clip_ablation_weights:
        cfg["clip_ablation_weights"] = [
            float(x) for x in str(args.clip_ablation_weights).split(",") if x.strip()
        ]
    run_pipeline(cfg)


if __name__ == "__main__":
    main()