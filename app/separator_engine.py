import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

logger = logging.getLogger("separator_engine")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Stems are kept lossless internally; MP3 is produced only for browser previews and exports.
MASTER_EXT = ".flac"

ProgressCallback = Callable[[float, str], None]


class SeparationCancelled(Exception):
    """Raised from a progress callback to abort a running separation."""


class _ProgressIterable:
    """Stands in for tqdm inside audio-separator so every model chunk reports progress.

    The callback runs before each chunk, so raising SeparationCancelled from it stops
    inference within one chunk instead of after the whole stage.
    """

    def __init__(self, iterable: Iterable, on_step: Callable[[int, int], None]):
        self._items = list(iterable)
        self._on_step = on_step

    def __iter__(self):
        total = len(self._items)
        for index, item in enumerate(self._items):
            self._on_step(index, total)
            yield item
        self._on_step(total, total)

    def __len__(self):
        return len(self._items)


@contextmanager
def _hook_chunk_progress(on_step: Callable[[int, int], None]):
    """Temporarily route audio-separator's MDXC/RoFormer chunk loop through on_step."""
    try:
        from audio_separator.separator.architectures import mdxc_separator
    except ImportError:
        yield
        return
    original = getattr(mdxc_separator, "tqdm", None)
    if original is None:
        logger.warning("audio-separator internals changed; chunk progress is unavailable.")
        yield
        return
    mdxc_separator.tqdm = lambda iterable, *args, **kwargs: _ProgressIterable(iterable, on_step)
    try:
        yield
    finally:
        mdxc_separator.tqdm = original


def _scaled(progress_cb: Optional[ProgressCallback], start: float, end: float, message: str):
    """Map a 0..1 fraction from a model loop onto the [start, end] progress range."""
    last = {"pct": -1}

    def report(fraction: float) -> None:
        if not progress_cb:
            return
        pct = int(start + (end - start) * max(0.0, min(1.0, fraction)))
        # Always call through (the callback is also the cancellation checkpoint).
        progress_cb(max(pct, last["pct"]), message)
        last["pct"] = max(pct, last["pct"])

    return report


class DualModeSeparatorEngine:
    """
    Dual-mode audio stem separator engine:
    1) 2 Stems: BS-RoFormer separates the mix into Vocals and Instrumental.
    2) 4 Stems: BS-RoFormer isolates Vocals and Instrumental, then Demucs splits
       ONLY the instrumental into Drums, Bass and Other (no vocal leakage).

    Models are cached on the instance so consecutive jobs don't reload weights.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.roformer_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        self._roformer_separator = None
        self._demucs_separator = None

    def _run_roformer(
        self,
        input_audio: Path,
        work_dir: Path,
        progress_cb: Optional[ProgressCallback],
        start: float,
        end: float,
    ) -> Dict[str, Path]:
        """Separate the mix into {"vocals": Path, "instrumental": Path} FLAC files."""
        from audio_separator.separator import Separator
        import torch
        from pydub import AudioSegment

        from app.media_tools import find_ffmpeg

        # pydub writes the output files; point it at the same ffmpeg the app uses
        # instead of whatever "ffmpeg" happens to be on PATH.
        AudioSegment.converter = find_ffmpeg()

        use_cuda = torch.cuda.is_available()
        logger.info(
            "BS-RoFormer device: %s%s",
            "cuda" if use_cuda else "cpu",
            f" ({torch.cuda.get_device_name(0)})" if use_cuda else "",
        )
        if progress_cb:
            progress_cb(start, "Loading AI separation model...")

        work_dir.mkdir(parents=True, exist_ok=True)
        if self._roformer_separator is None:
            self._roformer_separator = Separator(
                log_level=logging.INFO,
                model_file_dir=str(self.models_dir),
                output_dir=str(work_dir),
                output_format=MASTER_EXT.lstrip(".").upper(),
                sample_rate=44100,
                use_autocast=use_cuda,
            )
            self._roformer_separator.load_model(self.roformer_model)
        else:
            self._roformer_separator.output_dir = str(work_dir)
            if self._roformer_separator.model_instance is not None:
                self._roformer_separator.model_instance.output_dir = str(work_dir)
        separator = self._roformer_separator

        report = _scaled(progress_cb, start, end, "Separating vocals and instrumental...")
        report(0.0)
        with _hook_chunk_progress(lambda done, total: report(done / total if total else 1.0)):
            output_files = separator.separate(
                str(input_audio),
                custom_output_names={"Vocals": "vocals", "Instrumental": "instrumental"},
            )

        stems: Dict[str, Path] = {}
        candidates = [work_dir / Path(name).name for name in output_files or []]
        candidates += list(work_dir.glob(f"*{MASTER_EXT}"))
        for path in candidates:
            if not path.is_file():
                continue
            lower = path.name.lower()
            if "instrumental" in lower or "no_vocals" in lower:
                key = "instrumental"
            elif "vocal" in lower:
                key = "vocals"
            else:
                continue
            target = work_dir / f"{key}{MASTER_EXT}"
            if path != target:
                shutil.move(str(path), str(target))
            stems[key] = target

        if "vocals" not in stems or "instrumental" not in stems:
            raise RuntimeError(
                f"BS-RoFormer failed to generate required stems. Found: {list(work_dir.glob('*'))}"
            )
        return stems

    def _run_demucs_on_instrumental(
        self,
        instrumental_audio: Path,
        work_dir: Path,
        progress_cb: Optional[ProgressCallback],
        start: float,
        end: float,
    ) -> Dict[str, Path]:
        """Split the isolated instrumental into drums, bass and other FLAC files."""
        import soundfile as sf
        import torch
        from demucs.api import Separator
        from demucs.audio import prevent_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb:
            progress_cb(start, "Loading instrument separation model...")
        if self._demucs_separator is None:
            self._demucs_separator = Separator(
                model="htdemucs",
                device=device,
                jobs=4 if device == "cuda" else 2,
                progress=False,
            )
        demucs = self._demucs_separator

        report = _scaled(progress_cb, start, end, "Separating drums, bass and instruments...")

        def demucs_progress(info: dict) -> None:
            # Called at the start and end of every segment. Exceptions raised here
            # (cancellation) propagate and cancel the remaining segments.
            audio_length = info.get("audio_length") or 0
            models = info.get("models") or 1
            within = min(1.0, info.get("segment_offset", 0) / audio_length) if audio_length else 0.0
            report((info.get("model_idx_in_bag", 0) + within) / models)

        demucs.update_parameter(callback=demucs_progress)
        logger.info("Running cached Demucs model on instrumental using %s", device)
        report(0.0)
        _, separated = demucs.separate_audio_file(instrumental_audio)

        stems: Dict[str, Path] = {}
        for stem_name in ("drums", "bass", "other"):
            if stem_name not in separated:
                raise RuntimeError(f"Demucs did not return the {stem_name} stem.")
            wav = prevent_clip(separated[stem_name], mode="rescale")
            dst = work_dir / f"{stem_name}{MASTER_EXT}"
            # soundfile writes FLAC directly, so this stage doesn't need ffmpeg on PATH.
            sf.write(str(dst), wav.cpu().numpy().T, demucs.samplerate, subtype="PCM_24", format="FLAC")
            stems[stem_name] = dst
        if progress_cb:
            progress_cb(end, "Instrument stems written.")
        return stems

    def separate_2stems(
        self,
        input_audio: Path,
        output_dir: Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Path]:
        """BS-RoFormer only: Vocals + Instrumental."""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting 2-stem separation for %s", input_audio.name)
        stems = self._run_roformer(input_audio, output_dir, progress_cb, start=5, end=90)
        return {"vocals": stems["vocals"], "instrumental": stems["instrumental"]}

    def separate_4stems(
        self,
        input_audio: Path,
        output_dir: Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Path]:
        """BS-RoFormer for vocals, then Demucs on the instrumental for drums/bass/other."""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting 4-stem two-stage separation for %s", input_audio.name)
        roformer = self._run_roformer(input_audio, output_dir, progress_cb, start=5, end=55)
        instrumental = roformer["instrumental"]
        demucs = self._run_demucs_on_instrumental(instrumental, output_dir, progress_cb, start=55, end=90)
        # The intermediate instrumental isn't a 4-stem output; don't leave it to be served.
        instrumental.unlink(missing_ok=True)
        return {
            "vocals": roformer["vocals"],
            "drums": demucs["drums"],
            "bass": demucs["bass"],
            "other": demucs["other"],
        }
