"""Shared helpers used by every stage. Keep this file model-agnostic."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import config
import model


def run_dir() -> Path:
    return config.WORK_DIR.resolve() / config.RUN_ID


def adapter_dir() -> Path:
    return run_dir() / "adapter"


def heldout_path() -> Path:
    return adapter_dir() / "heldout.jsonl"


def reference_dir() -> Path:
    return run_dir() / "reference"


def reference_path() -> Path:
    return reference_dir() / "kuza-bf16.gguf"


def imatrix_dir() -> Path:
    return run_dir() / "imatrix"


def imatrix_path() -> Path:
    return imatrix_dir() / "kuza.imatrix"


def eval_path() -> Path:
    return imatrix_dir() / "eval.txt"


def quants_dir() -> Path:
    return run_dir() / "quants"


def screen_dir() -> Path:
    return run_dir() / "screen"


def training_dir() -> Path:
    return run_dir() / "training"


def candidate_gguf(name: str) -> Path:
    spec = model.QUANT_CANDIDATES[name]
    return quants_dir() / name / spec["filename"]


def configure_hf_cache() -> None:
    config.HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(config.HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.HF_CACHE_DIR / "hub"))


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set in the environment")
    configure_hf_cache()
    return token


def require_file(path: Path, hint: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing required artifact {path}. {hint}")
    return path


def run(
    command: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    log_path: Path | None = None,
) -> str:
    cmd = [str(part) for part in command]
    print("+", " ".join(cmd), flush=True)
    if log_path is None and not capture:
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        if result.returncode:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
            )
        return ""
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with contextlib.ExitStack() as stack:
        log_file = (
            stack.enter_context(log_path.open("w", encoding="utf-8"))
            if log_path
            else None
        )
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            print(line, end="", flush=True)
            if capture:
                chunks.append(line)
        returncode = process.wait()
    output = "".join(chunks)
    if returncode:
        raise RuntimeError(
            f"Command failed with exit code {returncode}: {' '.join(cmd)}"
        )
    return output


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def package_version(distribution: str) -> str:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(distribution)
    return "missing"


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def cuda_wheel_tag() -> str:
    text = ""
    nvcc = shutil.which("nvcc")
    if nvcc:
        text = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        ).stdout
    match = re.search(r"release (\d+)\.(\d+)", text)
    if not match:
        return "cu128"
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) >= (13, 0):
        return "cu130"
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (12, 4):
        return "cu124"
    return "cu128"


def pytorch_index_url() -> str:
    return f"https://download.pytorch.org/whl/{cuda_wheel_tag()}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_cuda_arch() -> str:
    if config.CUDA_ARCH:
        return str(config.CUDA_ARCH)
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return "80"


def llama_cpp_binaries() -> dict[str, Path]:
    checkout = config.TOOLS_DIR / "llama.cpp"
    build = checkout / "build"
    targets = [
        "llama-cli",
        "llama-imatrix",
        "llama-quantize",
        "llama-perplexity",
        "llama-bench",
    ]
    binaries = {name: build / "bin" / name for name in targets}
    binaries["converter"] = checkout / "convert_hf_to_gguf.py"
    missing = [str(path) for path in binaries.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"llama.cpp tools missing: {missing}. Run 01_setup_llama_cpp.py first."
        )
    return binaries


def setup_llama_cpp() -> dict[str, Path]:
    config.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    checkout = config.TOOLS_DIR / "llama.cpp"
    build = checkout / "build"
    targets = [
        "llama-cli",
        "llama-imatrix",
        "llama-quantize",
        "llama-perplexity",
        "llama-bench",
    ]
    binaries = {name: build / "bin" / name for name in targets}
    binaries["converter"] = checkout / "convert_hf_to_gguf.py"
    if checkout.exists() and all(path.exists() for path in binaries.values()):
        actual = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).strip()
        if actual == config.LLAMA_CPP_COMMIT:
            return binaries
    if not checkout.exists():
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", checkout])
    run(["git", "fetch", "origin", config.LLAMA_CPP_COMMIT], cwd=checkout)
    run(["git", "checkout", "--detach", config.LLAMA_CPP_COMMIT], cwd=checkout)
    actual = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).strip()
    if actual != config.LLAMA_CPP_COMMIT:
        raise RuntimeError(f"llama.cpp checkout mismatch: {actual}")
    cuda_arch = detect_cuda_arch()
    run(
        [
            "cmake",
            "-S",
            checkout,
            "-B",
            build,
            "-DGGML_NATIVE=OFF",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "-DGGML_CUDA_F16=ON",
            "-DGGML_CUDA_FA_ALL_QUANT=ON",
            "-DGGML_CCACHE=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            build,
            "--config",
            "Release",
            "-j",
            str(os.cpu_count() or 8),
            "--target",
            *targets,
        ]
    )
    missing = [str(path) for path in binaries.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing llama.cpp tools after build: {missing}")
    return binaries


def resolve_base_revision() -> str:
    from huggingface_hub import HfApi

    return HfApi(token=hf_token()).model_info(
        model.BASE_MODEL, revision=model.BASE_REVISION, token=hf_token()
    ).sha


def resolve_dataset_revisions() -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token())
    revisions: dict[str, str] = {}
    seen: dict[str, str] = {}
    for key, repo in model.DATASETS.items():
        root = config.LOCAL_DATA_DIR
        if root is not None:
            local = root / f"{key}.jsonl"
            if local.is_file() and local.stat().st_size > 0:
                revisions[key] = "local"
                continue
        if repo in seen:
            revisions[key] = seen[repo]
            continue
        sha = api.dataset_info(repo, revision="main", token=hf_token()).sha
        seen[repo] = sha
        revisions[key] = sha
    return revisions


def first_split(dataset_dict: Any) -> Any:
    if "train" in dataset_dict:
        return dataset_dict["train"]
    return dataset_dict[next(iter(dataset_dict.keys()))]


def content_from_turn(turn: dict[str, Any]) -> str:
    value = turn.get("content", turn.get("value", ""))
    if isinstance(value, list):
        value = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value).strip()


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    turns = row.get("messages") or row.get("conversations")
    cleaned: list[dict[str, str]] = []
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", turn.get("from", ""))).lower()
            if role in {"human"}:
                role = "user"
            if role in {"model", "gpt"}:
                role = "assistant"
            content = content_from_turn(turn)
            if role in {"user", "assistant"} and content:
                cleaned.append({"role": role, "content": content})
    if cleaned:
        return cleaned
    if row.get("instruction") and row.get("response"):
        return [
            {"role": "user", "content": str(row["instruction"]).strip()},
            {"role": "assistant", "content": str(row["response"]).strip()},
        ]
    if row.get("user") and row.get("assistant"):
        return [
            {"role": "user", "content": str(row["user"]).strip()},
            {"role": "assistant", "content": str(row["assistant"]).strip()},
        ]
    return None


def extract_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    messages = extract_messages(row)
    if not messages:
        return None
    user = ""
    assistant = ""
    for turn in messages:
        if turn["role"] == "user":
            user = turn["content"]
        elif turn["role"] == "assistant" and user:
            assistant = turn["content"]
    if user and assistant:
        return user, assistant
    return None


def rows_from_dataset(dataset: Any, language: str, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        pair = extract_pair(row)
        if not pair:
            continue
        instruction, response = pair
        messages = extract_messages(row)
        item: dict[str, Any] = {
            "instruction": instruction,
            "response": response,
            "language": str(row.get("language") or language),
            "source": str(row.get("source") or source),
            "source_id": str(row.get("id", row.get("source_id", f"{source}-{index}"))),
        }
        if messages:
            item["messages"] = messages
        rows.append(item)
    return rows


def rows_from_jsonl(path: Path, language: str, source: str) -> list[dict[str, str]]:
    return rows_from_dataset(read_jsonl(path), language, source)


def deterministic_sample(
    rows: Sequence[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[: min(count, len(rows))]


def split_rows(
    rows: Sequence[dict[str, str]], test_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    return shuffled[test_count:], shuffled[:test_count]


def load_hub_rows(
    key: str,
    language: str,
    dataset_revisions: dict[str, str],
) -> list[dict[str, str]]:
    from datasets import load_dataset

    repo = model.DATASETS[key]
    revision = dataset_revisions.get(key, "main")
    if revision == "local":
        revision = "main"
    token = hf_token()
    try:
        loaded = load_dataset(repo, split="train", revision=revision, token=token)
    except Exception:
        try:
            loaded = first_split(
                load_dataset(repo, revision=revision, token=token)
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load {key} from {repo}: {exc}") from exc
    return rows_from_dataset(loaded, language, key)


def load_local_or_hub(
    key: str,
    language: str,
    filename: str,
    dataset_revisions: dict[str, str],
    *,
    hub_fallback: bool = True,
) -> list[dict[str, str]]:
    root = config.LOCAL_DATA_DIR
    if root is not None:
        local = root / filename
        if local.is_file() and local.stat().st_size > 0:
            return rows_from_jsonl(local, language, key)
    if not hub_fallback:
        return []
    return load_hub_rows(key, language, dataset_revisions)


def load_training_rows(
    dataset_revisions: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    english = load_local_or_hub(
        "english", "english", "english.jsonl", dataset_revisions
    )
    if not english:
        raise RuntimeError(
            "No English rows. Push data/final/english.jsonl to "
            f"{model.DATASETS['english']} or set KUZA_LOCAL_DATA."
        )
    swahili = load_local_or_hub(
        "swahili",
        "swahili",
        "swahili.jsonl",
        dataset_revisions,
    )
    adversarial = load_local_or_hub(
        "adversarial", "english", "adversarial.jsonl", dataset_revisions
    )
    general = load_local_or_hub(
        "general", "english", "general.jsonl", dataset_revisions
    )
    general = [
        row
        for row in general
        if len(row["instruction"].split()) <= model.MIX["general_max_instruction_words"]
        and len(row["response"].split()) <= model.MIX["general_max_response_words"]
    ]
    multiturn = load_local_or_hub(
        "multiturn", "english", "multiturn.jsonl", dataset_revisions
    )

    en_train, en_eval = split_rows(
        english, model.MIX["agri_eval_fraction"], config.SEED
    )
    sw_train, sw_eval = split_rows(
        swahili, model.MIX["agri_eval_fraction"], config.SEED
    ) if swahili else ([], [])
    adv_train, adv_eval = split_rows(
        adversarial, model.MIX["adversarial_eval_fraction"], config.SEED
    ) if adversarial else ([], [])

    sw_target = round(len(en_train) * model.MIX["swahili_of_english"])
    if sw_target > 0 and not sw_train:
        raise RuntimeError(
            "MIX requests Swahili "
            f"({model.MIX['swahili_of_english']:.0%} of English) but no "
            "swahili rows were found. Translate, run data/prepare_swahili.py, "
            f"push to {model.DATASETS['swahili']}, and set KUZA_SW_DATASET."
        )
    general_target = round(len(en_train) * model.MIX["general_of_english"])
    if general_target > 0 and not general:
        raise RuntimeError(
            "MIX requests general "
            f"({model.MIX['general_of_english']:.0%} of English) but no "
            "general rows were found. Set KUZA_GENERAL_DATASET "
            f"(default {model.DATASETS['general']})."
        )
    adversarial_target = round(len(en_train) * model.MIX["adversarial_of_english"])
    if adversarial_target > 0 and not adv_train:
        raise RuntimeError(
            "MIX requests adversarial "
            f"({model.MIX['adversarial_of_english']:.0%} of English) but no "
            "adversarial rows were found. Push data/final/adversarial.jsonl "
            f"to {model.DATASETS['adversarial']} or set KUZA_ADV_DATASET."
        )
    if not multiturn:
        raise RuntimeError(
            "No multiturn rows. Push data/final/multiturn.jsonl to "
            f"{model.DATASETS['multiturn']} or set KUZA_MT_DATASET."
        )
    train = [
        *en_train,
        *deterministic_sample(sw_train, sw_target, config.SEED + 1),
        *deterministic_sample(general, general_target, config.SEED + 2),
        *deterministic_sample(adv_train, adversarial_target, config.SEED + 3),
        *multiturn,
    ]
    random.Random(config.SEED).shuffle(train)

    eval_group_count = min(
        model.MIX["eval_group_max"],
        max(1, len(en_eval)),
        max(1, len(sw_eval) or 1),
        max(1, len(adv_eval) or 1),
    )
    evaluation = [
        *deterministic_sample(en_eval, eval_group_count, config.SEED + 4),
        *deterministic_sample(sw_eval, min(eval_group_count, len(sw_eval)), config.SEED + 5),
        *deterministic_sample(adv_eval, min(eval_group_count, len(adv_eval)), config.SEED + 6),
    ]
    random.Random(config.SEED).shuffle(evaluation)
    heldout = [*en_eval, *sw_eval, *deterministic_sample(general, 250, config.SEED + 7)]
    return train, evaluation, heldout


def contains_subsequence(values: Sequence[int], needle: Sequence[int]) -> bool:
    width = len(needle)
    return any(
        values[index : index + width] == list(needle)
        for index in range(len(values) - width + 1)
    )


def prepare_sft_data(
    tokenizer: Any,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    heldout_rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    from datasets import Dataset

    marker_ids = tokenizer(model.RESPONSE_PART, add_special_tokens=False)["input_ids"]

    def process(
        rows: list[dict[str, str]], split: str
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        valid: list[dict[str, str]] = []
        invalid = 0
        truncated = 0
        for row in rows:
            text = model.render_text(tokenizer, row)
            full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            is_truncated = len(full_ids) > config.MAX_SEQ_LENGTH
            # keep_start: never left-slice. A tail crop drops the system/user
            # turn and can still match RESPONSE_PART on the assistant answer.
            limited_ids = (
                full_ids[: config.MAX_SEQ_LENGTH] if is_truncated else full_ids
            )
            has_response = contains_subsequence(limited_ids, marker_ids)
            if is_truncated:
                truncated += 1
                text = tokenizer.decode(limited_ids, skip_special_tokens=False)
            if not has_response or len(limited_ids) <= len(marker_ids) + 1:
                invalid += 1
                continue
            valid.append({**row, "text": text})
        report = {
            "split": split,
            "input_rows": len(rows),
            "valid_rows": len(valid),
            "invalid_rows": invalid,
            "invalid_fraction": invalid / max(1, len(rows)),
            "truncated_rows": truncated,
            "truncated_fraction": truncated / max(1, len(rows)),
            "composition": dict(Counter(row["source"] for row in valid)),
            "languages": dict(Counter(row["language"] for row in valid)),
        }
        if report["invalid_fraction"] > config.MAX_INVALID_FRACTION:
            raise RuntimeError(f"{split} invalid-label fraction exceeds limit: {report}")
        if report["truncated_fraction"] > config.MAX_TRUNCATED_FRACTION:
            raise RuntimeError(f"{split} truncation fraction exceeds limit: {report}")
        return valid, report

    train_valid, train_report = process(train_rows, "train")
    eval_valid, eval_report = process(eval_rows, "eval")
    quality = {"train": train_report, "eval": eval_report}
    write_json(output_dir / "data_quality.json", quality)
    write_jsonl(output_dir / "heldout.jsonl", heldout_rows)
    columns = ["text", "instruction", "response", "language", "source", "source_id"]
    if any("messages" in row for row in train_valid + eval_valid):
        for row in train_valid + eval_valid:
            row.setdefault("messages", [])
        columns.append("messages")
    return (
        Dataset.from_list(train_valid).select_columns(columns),
        Dataset.from_list(eval_valid).select_columns(columns),
        quality,
    )


def checkpoint_weight_keys(repo_id: str, revision: str) -> set[str]:
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open

    files = HfApi().list_repo_files(
        repo_id, revision=revision, token=hf_token()
    )
    indexes = [name for name in files if name.endswith(".safetensors.index.json")]
    if indexes:
        index_path = hf_hub_download(
            repo_id,
            indexes[0],
            revision=revision,
            token=hf_token(),
        )
        return set(read_json(Path(index_path))["weight_map"])
    safetensor_files = [name for name in files if name.endswith(".safetensors")]
    keys: set[str] = set()
    for name in safetensor_files:
        path = hf_hub_download(
            repo_id, name, revision=revision, token=hf_token()
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def assert_model_bf16(loaded: Any, label: str) -> None:
    import torch

    counts = Counter(
        str(parameter.dtype)
        for parameter in loaded.parameters()
        if parameter.is_floating_point()
    )
    forbidden = {
        dtype: count for dtype, count in counts.items() if dtype != str(torch.bfloat16)
    }
    if forbidden:
        raise RuntimeError(f"{label} contains non-BF16 floating weights: {forbidden}")


def adapter_weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("adapter_model*.safetensors"))
    if not files:
        raise RuntimeError(f"No adapter_model*.safetensors found in {directory}")
    return files


def adapter_file_keys(directory: Path) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for path in adapter_weight_files(directory):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    lora_keys = {key for key in keys if "lora_" in key}
    if not lora_keys:
        raise RuntimeError(f"Saved adapter contains no LoRA tensors: {sorted(keys)[:20]}")
    return lora_keys


def normalize_lora_key(key: str) -> str:
    key = re.sub(r"\.default(?=\.weight$)", "", key)
    key = key.replace(".linear.lora_", ".lora_")
    for prefix in ("base_model.model.", "base_model.", "model."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def assert_adapter_tensors_loaded(loaded: Any, directory: Path) -> None:
    saved = {normalize_lora_key(key) for key in adapter_file_keys(directory)}
    attached = {
        normalize_lora_key(name)
        for name, _ in loaded.named_parameters()
        if "lora_" in name
    }
    missing = sorted(saved - attached)
    unexpected = sorted(attached - saved)
    if missing or unexpected:
        raise RuntimeError(
            "Loaded PEFT adapter keys do not match the saved adapter weights. "
            "This usually means Unsloth module names (for example q_proj.linear) "
            "did not attach to the Transformers merge model. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def inspect_safetensors_bf16(directory: Path) -> dict[str, int]:
    from safetensors import safe_open

    counts: Counter[str] = Counter()
    for path in directory.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = str(handle.get_slice(key).get_dtype())
                counts[dtype] += 1
                if dtype != "BF16":
                    raise RuntimeError(f"Non-BF16 merged tensor {key} has dtype {dtype}")
    if not counts or counts.get("BF16", 0) == 0:
        raise RuntimeError(f"No BF16 tensors found in {directory}")
    return dict(counts)


def gguf_inventory(path: Path, llama_cpp_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(llama_cpp_root / "gguf-py"))
    try:
        from gguf import GGMLQuantizationType, GGUFReader

        reader = GGUFReader(str(path), "r")
        counts: Counter[str] = Counter()
        tensors: dict[str, str] = {}
        for tensor in reader.tensors:
            dtype = GGMLQuantizationType(tensor.tensor_type).name
            counts[dtype] += 1
            tensors[tensor.name] = dtype
        architecture = None
        field = reader.fields.get("general.architecture")
        if field is not None:
            architecture = str(field.parts[-1])
        return {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "architecture": architecture,
            "tensor_type_counts": dict(counts),
            "tensors": tensors,
        }
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(llama_cpp_root / "gguf-py"))


def smoke_load(binary: Path, weights: Path, prompt: str, log_path: Path) -> str:
    command = [
        binary,
        "-m",
        weights,
        "-p",
        prompt,
        "-n",
        "24",
        "-c",
        str(config.MAX_SEQ_LENGTH),
        "-ngl",
        "999",
        "--temp",
        "0",
        "--no-display-prompt",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "-fa",
        str(getattr(config, "FLASH_ATTN", "on")),
        "-ctk",
        str(getattr(config, "CACHE_TYPE_K", "q8_0")),
        "-ctv",
        str(getattr(config, "CACHE_TYPE_V", "q8_0")),
    ]
    extra = getattr(config, "SMOKE_EXTRA_ARGS", ())
    if extra:
        command.extend(str(part) for part in extra)
    return run(command, capture=True, log_path=log_path)


REQUIRED_SFT_CONFIG_KEYS = (
    "bf16",
    "fp16",
    "lr_scheduler_type",
    "warmup_ratio",
    "seed",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
)


def filter_sft_config(
    kwargs: dict[str, Any],
    parameters: Iterable[str],
    *,
    length_key: str,
    max_seq_length: int,
) -> dict[str, Any]:
    parameter_set = set(parameters)
    filtered = {key: value for key, value in kwargs.items() if key in parameter_set}
    required = (*REQUIRED_SFT_CONFIG_KEYS, length_key)
    missing = [key for key in required if key not in filtered]
    if missing:
        raise RuntimeError(f"SFTConfig dropped required keys: {missing}")
    if filtered.get("bf16") is not True or filtered.get("fp16") is not False:
        raise RuntimeError(
            f"SFTConfig lost the BF16 gate: bf16={filtered.get('bf16')!r}, "
            f"fp16={filtered.get('fp16')!r}"
        )
    if filtered.get(length_key) != max_seq_length:
        raise RuntimeError(
            f"SFTConfig {length_key}={filtered.get(length_key)!r} != {max_seq_length}"
        )
    return filtered


def assert_completion_only_labels(
    trainer: Any,
    sample_size: int = 64,
) -> dict[str, float]:
    dataset = getattr(trainer, "train_dataset", None)
    collator = getattr(trainer, "data_collator", None)
    if dataset is None or collator is None or len(dataset) == 0:
        raise RuntimeError("Cannot inspect completion-only labels before training")
    count = min(sample_size, len(dataset))
    ignored = 0
    for index in range(count):
        batch = collator([dataset[index]])
        labels = batch.get("labels")
        if labels is None:
            ignored += 1
            continue
        if hasattr(labels, "numel") and labels.numel() > 0 and bool((labels == -100).all()):
            ignored += 1
        elif hasattr(labels, "__iter__") and not hasattr(labels, "numel"):
            flat = list(labels[0] if labels and hasattr(labels[0], "__iter__") else labels)
            if flat and all(int(value) == -100 for value in flat):
                ignored += 1
    fraction = ignored / max(1, count)
    report = {"sampled": count, "all_ignored": ignored, "all_ignored_fraction": fraction}
    if fraction > config.MAX_INVALID_FRACTION:
        raise RuntimeError(f"Completion-only all -100 label fraction exceeds limit: {report}")
    return report


def preserve_best_checkpoint(output_dir: Path, best_checkpoint: str | None) -> Path:
    destination = output_dir / "checkpoint-best"
    if best_checkpoint:
        source = Path(best_checkpoint)
        if source.exists() and source.resolve() != destination.resolve():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
    if not destination.exists():
        raise RuntimeError("Best checkpoint is missing after training")
    return destination


def select_balanced(
    rows: Sequence[dict[str, Any]], per_language: int, seed: int
) -> list[dict[str, Any]]:
    english = [row for row in rows if row.get("language") == "english"]
    swahili = [row for row in rows if row.get("language") == "swahili"]
    return [
        *deterministic_sample(english, per_language, seed),
        *deterministic_sample(swahili, per_language, seed + 1),
    ]


def select_mixcal(
    rows: Sequence[dict[str, Any]],
    agri_per_language: int,
    generic_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    english = [
        row
        for row in rows
        if row.get("language") == "english" and row.get("source") != "general"
    ]
    swahili = [row for row in rows if row.get("language") == "swahili"]
    generic = [row for row in rows if row.get("source") == "general"]
    if not generic:
        generic = [
            row
            for row in rows
            if row.get("language") == "english" and row.get("source") == "english"
        ]
    return [
        *deterministic_sample(english, agri_per_language, seed),
        *deterministic_sample(swahili, agri_per_language, seed + 1),
        *deterministic_sample(generic, generic_count, seed + 2),
    ]


def render_corpus(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    output: Path,
    max_seq_length: int,
) -> dict[str, Any]:
    rendered_rows: list[str] = []
    truncated = 0
    total_tokens = 0
    for row in rows:
        text = model.render_text(tokenizer, row)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_seq_length:
            ids = ids[:max_seq_length]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            truncated += 1
        total_tokens += len(ids)
        rendered_rows.append(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(rendered_rows) + "\n", encoding="utf-8")
    return {
        "rows": len(rendered_rows),
        "truncated": truncated,
        "tokens": total_tokens,
        "sha256": sha256_file(output),
        "size": output.stat().st_size,
    }


def tool_help(binary: Path) -> str:
    last = ""
    for flag in ("-h", "--help"):
        try:
            return run([binary, flag], capture=True)
        except RuntimeError as exc:
            last = str(exc)
    return last


def help_has(help_text: str, flag: str) -> bool:
    return re.search(rf"(?:^|\s){re.escape(flag)}(?:\s|,|$)", help_text) is not None


def generation_tps(bench_values: Sequence[float]) -> float:
    return float(bench_values[-1]) if bench_values else 0.0


def parse_mean_kld(output: str) -> float | None:
    patterns = [
        r"Mean KLD[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        r"mean kl divergence[^0-9]*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_bench_tps(output: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:±[^|]*)?\|\s*$", output, re.MULTILINE
        )
    ]


def installed_packages() -> dict[str, str]:
    names = [*config.PINNED_PACKAGES, "torch", "accelerate", "huggingface-hub"]
    return {name: package_version(name) for name in names}
