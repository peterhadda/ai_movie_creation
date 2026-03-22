from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.io_utils import save_evaluation_report
from src.common.logging_utils import log_message
from src.phase6_multimodal.audio.generator import generate_audio
from src.phase6_multimodal.fusion.alignment import align_modalities, combine_embeddings
from src.phase6_multimodal.image.generator import generate_image
from src.phase6_multimodal.text.encoder import encode_text
from src.phase6_multimodal.video.generator import generate_video


def run_multimodal_pipeline(config_path: str | Path = "configs/phase6.json") -> dict[str, Any]:
    config = load_config(config_path)["phase6"]
    text_input = config["text_input"]

    log_message("Encoding text prompt")
    text_embedding = encode_text(text_input, embedding_dim=config.get("embedding_dim", 64))
    shared_embedding = combine_embeddings([text_embedding])

    image_alignment = align_modalities(text_embedding, shared_embedding)
    audio_alignment = align_modalities(text_embedding, shared_embedding)
    video_alignment = align_modalities(text_embedding, shared_embedding)

    log_message("Generating image output")
    generated_image = generate_image(
        shared_embedding,
        model_path=config["models"]["image_model"],
        output_path=config["outputs"]["image"],
    )

    log_message("Generating audio output")
    generated_audio = generate_audio(
        shared_embedding,
        model_path=config["models"]["audio_model"],
        output_path=config["outputs"]["audio"],
        sample_rate=config.get("audio_sample_rate", 16000),
        duration=config.get("audio_duration", 1.0),
    )

    log_message("Generating video output")
    generated_video = generate_video(
        shared_embedding,
        model_path=config["models"]["video_model"],
        output_path=config["outputs"]["video"],
        fps=config.get("video_fps", 8),
    )

    report = {
        "text_input": text_input,
        "image_output": str(generated_image),
        "audio_output": str(generated_audio),
        "video_output": str(generated_video),
        "alignment_scores": {
            "image": image_alignment,
            "audio": audio_alignment,
            "video": video_alignment,
        },
    }
    report_path = save_evaluation_report(report, config["outputs"]["report"])

    return {
        "text_embedding": text_embedding,
        "shared_embedding": shared_embedding,
        "generated_image": generated_image,
        "generated_audio": generated_audio,
        "generated_video": generated_video,
        "report_path": report_path,
    }
