"""Whisper model catalog, install checks, and downloads for Sonex."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

# (model_id, tooltip description)
WHISPER_MODEL_OPTIONS: List[Tuple[str, str]] = [
    ("tiny", "speed"),
    ("base", "bare minimum"),
    ("small", "not recommended for normal users"),
    ("medium", "not recommended for normal users"),
    ("large-v2", "not recommended for normal users"),
    ("large-v3", "accuracy"),
    ("large-v3-turbo", "recommended for laptops"),
]

WHISPER_MODEL_IDS = {model_id for model_id, _ in WHISPER_MODEL_OPTIONS}

_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def normalize_whisper_model_name(value: Optional[str], default: str = "medium") -> str:
    """Map stored settings (including legacy labeled values) to a model id."""
    if not value:
        return default

    model_id = str(value).strip()
    if model_id.startswith("(!) "):
        model_id = model_id[4:].strip()
    if " (" in model_id:
        model_id = model_id.split(" (", 1)[0].strip()

    if model_id in WHISPER_MODEL_IDS:
        return model_id
    return default


def is_whisper_model_installed(model_id: str) -> bool:
    from faster_whisper import download_model

    try:
        download_model(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def whisper_model_display_name(model_id: str, installed: Optional[bool] = None) -> str:
    if installed is None:
        installed = is_whisper_model_installed(model_id)
    if installed:
        return model_id
    return f"(!) {model_id}"


class _ByteDownloadProgress:
    """Track byte-level progress across sequential Hugging Face file downloads."""

    def __init__(self, progress_cb: Callable[[int, str], None], model_id: str):
        self.progress_cb = progress_cb
        self.model_id = model_id
        self._active: Dict[int, object] = {}
        self._completed_bytes = 0
        self._last_pct = -1
        self._last_file = ""

    def emit(self, force: bool = False):
        in_flight = sum(getattr(bar, "n", 0) for bar in self._active.values())
        in_flight_total = sum(getattr(bar, "total", 0) or 0 for bar in self._active.values())
        done = self._completed_bytes + in_flight
        total = self._completed_bytes + in_flight_total

        current_file = ""
        for bar in self._active.values():
            current_file = getattr(bar, "desc", None) or current_file

        if total > 0:
            pct = min(99, int(100 * done / total))
            label = f"Downloading {self.model_id}: {current_file} ({pct}%)"
        else:
            pct = 0
            label = f"Downloading {self.model_id}: {current_file or 'starting...'}"

        if not force and pct == self._last_pct and current_file == self._last_file:
            return

        self._last_pct = pct
        self._last_file = current_file
        self.progress_cb(pct, label)

    def make_tqdm_class(self, base_tqdm):
        reporter = self

        class _ReportingTqdm(base_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                reporter._active[id(self)] = self
                reporter.emit()

            def update(self, n=1):
                super().update(n)
                reporter.emit()

            def close(self):
                total = getattr(self, "total", None)
                if total:
                    reporter._completed_bytes += int(total)
                reporter._active.pop(id(self), None)
                super().close()
                reporter.emit(force=True)

        return _ReportingTqdm


def _list_repo_files(repo_id: str) -> Tuple[str, List[str]]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi()
    repo_info = api.repo_info(repo_id, repo_type="model")
    all_files = [sibling.rfilename for sibling in repo_info.siblings]
    filtered = list(
        filter_repo_objects(
            items=all_files,
            allow_patterns=_ALLOW_PATTERNS,
        )
    )
    return repo_info.sha, filtered


def download_whisper_model(
    model_id: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Download a Whisper model to the Hugging Face cache."""
    import importlib

    from faster_whisper.utils import _MODELS
    import huggingface_hub.file_download as hf_file_download
    from huggingface_hub import hf_hub_download

    hf_tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")

    repo_id = _MODELS.get(model_id)
    if repo_id is None:
        raise ValueError(
            f"Invalid whisper model '{model_id}', expected one of: {', '.join(sorted(WHISPER_MODEL_IDS))}"
        )

    if progress_cb is None:
        from faster_whisper import download_model

        return download_model(model_id)

    revision, repo_files = _list_repo_files(repo_id)
    if not repo_files:
        raise RuntimeError(f"No downloadable files found for {repo_id}")

    progress_cb(0, f"Downloading {model_id}...")

    reporter = _ByteDownloadProgress(progress_cb, model_id)
    original_hf_tqdm = hf_tqdm_module.tqdm
    original_file_tqdm = hf_file_download.tqdm
    reporting_tqdm = reporter.make_tqdm_class(original_hf_tqdm)

    hf_tqdm_module.tqdm = reporting_tqdm
    hf_file_download.tqdm = reporting_tqdm

    last_path = None
    try:
        for repo_file in repo_files:
            last_path = hf_hub_download(
                repo_id,
                filename=repo_file,
                revision=revision,
            )
    finally:
        hf_tqdm_module.tqdm = original_hf_tqdm
        hf_file_download.tqdm = original_file_tqdm

    progress_cb(100, f"Downloaded {model_id}")
    return last_path or ""
