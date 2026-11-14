import os
import sys
import re
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger("separator_engine")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VENV_SCRIPTS = Path(sys.executable).parent
DEMUCS_EXE = VENV_SCRIPTS / "demucs.exe"
if not DEMUCS_EXE.exists():
    DEMUCS_EXE = "demucs"

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

        if progress_cb:
            progress_cb(10, "Initializing BS-RoFormer AI model...")

        work_dir.mkdir(parents=True, exist_ok=True)

        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=str(self.models_dir),
            output_dir=str(work_dir),
            output_format="MP3",
            output_bitrate="256k",
            sample_rate=44100
        )

        if progress_cb:
            progress_cb(18, "Loading BS-RoFormer neural weights (SDR 12.97)...")
        separator.load_model(self.roformer_model)

        if progress_cb:
            progress_cb(25, "BS-RoFormer is separating vocals and instrumental stems...")

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
            progress_cb(55, "Stage 2/2: Initializing Demucs for instrumental decomposition...")

        demucs_out_dir = work_dir / "demucs_temp"
        demucs_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(DEMUCS_EXE),
            "-n", "htdemucs",
            "-d", "cpu",
            "-j", "2",
            "--mp3",
            "--mp3-bitrate", "256",
            "-o", str(demucs_out_dir),
            str(instrumental_audio)
        ]

        logger.info(f"Running Demucs on instrumental: {' '.join(cmd)}")
        if progress_cb:
            progress_cb(60, "Stage 2/2: Demucs is separating Drums, Bass & Instruments...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        pattern = re.compile(r'(\d+)%')
        buffer = ""
        while True:
            char = process.stderr.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                buffer += char
                if char in ['\r', '\n']:
                    match = pattern.search(buffer)
                    if match and progress_cb:
                        pct = int(match.group(1))
                        # Scale Demucs progress from 60% to 90%
                        scaled_pct = 60 + int((pct / 100.0) * 30)
                        progress_cb(scaled_pct, f"Stage 2/2: Demucs separating instruments ({pct}%)...")
                    buffer = ""

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Demucs execution failed (code {process.returncode}): {stderr[-300:]}")

        # Locate Demucs output folder
        stem_folder = demucs_out_dir / "htdemucs" / instrumental_audio.stem
        if not stem_folder.exists():
            candidates = list((demucs_out_dir / "htdemucs").glob("*"))
            if candidates and candidates[0].is_dir():
                stem_folder = candidates[0]
            else:
                raise FileNotFoundError(f"Demucs output folder not found in {demucs_out_dir}")

        stems = {}
        for stem_name in ["drums", "bass", "other"]:
            src = stem_folder / f"{stem_name}.mp3"
            if src.exists():
                dst = work_dir / f"{stem_name}.mp3"
                shutil.copy2(str(src), str(dst))
                stems[stem_name] = dst
            else:
                raise FileNotFoundError(f"Demucs stem {stem_name}.mp3 missing from {stem_folder}")

        # Clean up temporary demucs folder to save disk space
        try:
            shutil.rmtree(str(demucs_out_dir), ignore_errors=True)
        except Exception:
            pass

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
            progress_cb(92, "Finalizing 2-stem separation...")

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

        # Stage 1: BS-RoFormer
        if progress_cb:
            progress_cb(5, "Stage 1/2: Isolating clean vocals with BS-RoFormer...")

        roformer_stems = self._run_roformer(input_audio, output_dir, progress_cb)
        clean_vocals = roformer_stems["vocals"]
        clean_instrumental = roformer_stems["instrumental"]

        if progress_cb:
            progress_cb(50, "Stage 1/2 complete. Isolated instrumental ready for Demucs...")

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
            progress_cb(92, "Finalizing 4-stem pipeline and aligning tracks...")

        return final_stems
