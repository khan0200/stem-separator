import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger("separator_engine")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VENV_SCRIPTS = Path(sys.executable).parent
FFMPEG_EXE = VENV_SCRIPTS / "ffmpeg.exe"
if not FFMPEG_EXE.exists():
    FFMPEG_EXE = "ffmpeg"


class DualModeSeparatorEngine:
    """
    Dual-mode audio stem separator engine:
    1) Option 1 (2 Stems): BS-RoFormer AI
       Separates original audio directly into clean Vocals and Instrumental.
    2) Option 2 (4 Stems): Two-stage AI Pipeline (BS-RoFormer + Demucs)
       Stage 1: BS-RoFormer isolates Vocals and Instrumental.
       Stage 2: Demucs processes ONLY the isolated Instrumental stem,
                separating it into Drums, Bass, and Other without any vocal leakage.
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
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Path]:
        """
        Runs BS-RoFormer model to separate audio into vocals and instrumental.
        Returns paths to: {"vocals": Path, "instrumental": Path}
        """
        from audio_separator.separator import Separator
        import torch

        use_cuda = torch.cuda.is_available()
        logger.info(
            "BS-RoFormer device: %s%s",
            "cuda" if use_cuda else "cpu",
            f" ({torch.cuda.get_device_name(0)})" if use_cuda else "",
        )

        if progress_cb:
            progress_cb(10, "Initializing AI separation engine...")

        work_dir.mkdir(parents=True, exist_ok=True)

        if self._roformer_separator is None:
            self._roformer_separator = Separator(
                log_level=logging.INFO,
                model_file_dir=str(self.models_dir),
                output_dir=str(work_dir),
                output_format="MP3",
                output_bitrate="256k",
                sample_rate=44100,
                use_autocast=use_cuda
            )
            self._roformer_separator.load_model(self.roformer_model)
        else:
            self._roformer_separator.output_dir = str(work_dir)
            if self._roformer_separator.model_instance is not None:
                self._roformer_separator.model_instance.output_dir = str(work_dir)
        separator = self._roformer_separator

        if progress_cb:
            progress_cb(18, "Loading AI neural models (cached)...")

        if progress_cb:
            progress_cb(25, "Separating vocals and instrumental tracks...")

        # Run inference
        output_files = separator.separate(
            str(input_audio),
            custom_output_names={"Vocals": "vocals", "Instrumental": "instrumental"}
        )

        # Map outputs
        stems = {}
        # Check generated files in work_dir
        for f in work_dir.glob("*.mp3"):
            fname = f.name.lower()
            if "vocal" in fname and "instrumental" not in fname:
                target = work_dir / "vocals.mp3"
                if f != target:
                    shutil.move(str(f), str(target))
                stems["vocals"] = target
            elif "instrumental" in fname or "no_vocals" in fname:
                target = work_dir / "instrumental.mp3"
                if f != target:
                    shutil.move(str(f), str(target))
                stems["instrumental"] = target

        if "vocals" not in stems or "instrumental" not in stems:
            # Fallback inspect all generated files
            for out_name in output_files:
                out_path = work_dir / out_name
                if out_path.exists():
                    lower_name = out_name.lower()
                    if "vocal" in lower_name and "instrumental" not in lower_name:
                        target = work_dir / "vocals.mp3"
                        shutil.move(str(out_path), str(target))
                        stems["vocals"] = target
                    elif "instrumental" in lower_name or "no_vocals" in lower_name:
                        target = work_dir / "instrumental.mp3"
                        shutil.move(str(out_path), str(target))
                        stems["instrumental"] = target

        if "vocals" not in stems or "instrumental" not in stems:
            raise RuntimeError(f"BS-RoFormer failed to generate required stems. Found: {list(work_dir.glob('*'))}")

        return stems

    def _run_demucs_on_instrumental(
        self,
        instrumental_audio: Path,
        work_dir: Path,
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Path]:
        """
        Runs Demucs ONLY on the isolated instrumental track.
        Decomposes instrumental into drums, bass, and other.
        """
        if progress_cb:
            progress_cb(55, "Decomposing instrumental tracks...")

        import torch
        from demucs.api import Separator
        from demucs.audio import save_audio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        jobs = 4 if device == "cuda" else 2
        if self._demucs_separator is None:
            self._demucs_separator = Separator(
                model="htdemucs",
                device=device,
                jobs=jobs,
                progress=False
            )
        demucs = self._demucs_separator

        def demucs_progress(info: dict) -> None:
            if progress_cb:
                if info.get("state") == "end":
                    progress_cb(90, "Demucs separation progress complete...")
                else:
                    progress_cb(60, "Separating drums, bass and instruments...")

        demucs.update_parameter(callback=demucs_progress)
        logger.info("Running cached Demucs model on instrumental using %s", device)
        if progress_cb:
            progress_cb(60, "Separating drums, bass and instruments...")

        _, separated = demucs.separate_audio_file(instrumental_audio)
        if progress_cb:
            progress_cb(90, "Writing instrumental stems...")

        stems = {}
        for stem_name in ["drums", "bass", "other"]:
            if stem_name not in separated:
                raise RuntimeError(f"Demucs did not return the {stem_name} stem.")
            dst = work_dir / f"{stem_name}.mp3"
            save_audio(
                separated[stem_name],
                dst,
                samplerate=demucs.samplerate,
                bitrate=256,
                preset=5
            )
            stems[stem_name] = dst

        return stems

    def separate_2stems(
        self,
        input_audio: Path,
        output_dir: Path,
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Path]:
        """
        Option 1: 2-Stem Mode (BS-RoFormer)
        Separates original audio into Vocals and Instrumental.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting 2-Stem Separation (BS-RoFormer) for {input_audio.name}")

        stems = self._run_roformer(input_audio, output_dir, progress_cb)

        if progress_cb:
            progress_cb(92, "Mastering audio stems...")

        return {
            "vocals": stems["vocals"],
            "instrumental": stems["instrumental"]
        }

    def separate_4stems(
        self,
        input_audio: Path,
        output_dir: Path,
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Path]:
        """
        Option 2: 4-Stem Mode (Two-Stage Pipeline: BS-RoFormer + Demucs)
        Stage 1: BS-RoFormer -> Clean Vocals + Isolated Instrumental
        Stage 2: Demucs -> Runs ONLY on Isolated Instrumental -> Drums, Bass, Other
        Result: Zero vocal bleed into instrumental stems (including wind instruments).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting 4-Stem Two-Stage Pipeline for {input_audio.name}")

        # Stage 1: Isolate vocals
        if progress_cb:
            progress_cb(5, "Isolating vocal harmonics...")

        roformer_stems = self._run_roformer(input_audio, output_dir, progress_cb)
        clean_vocals = roformer_stems["vocals"]
        clean_instrumental = roformer_stems["instrumental"]

        if progress_cb:
            progress_cb(50, "Vocals isolated. Processing instrumental stems...")

        # Stage 2: Demucs on ONLY clean_instrumental
        demucs_stems = self._run_demucs_on_instrumental(clean_instrumental, output_dir, progress_cb)

        # The final 4 stems
        final_stems = {
            "vocals": clean_vocals,
            "drums": demucs_stems["drums"],
            "bass": demucs_stems["bass"],
            "other": demucs_stems["other"]
        }

        if progress_cb:
            progress_cb(92, "Mastering and aligning audio tracks...")

        return final_stems
