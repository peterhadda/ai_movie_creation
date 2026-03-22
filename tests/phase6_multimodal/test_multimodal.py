import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.phase6_multimodal.audio.generator import generate_audio
from src.phase6_multimodal.fusion.alignment import align_modalities, combine_embeddings
from src.phase6_multimodal.image.generator import generate_image
from src.phase6_multimodal.pipeline import run_multimodal_pipeline
from src.phase6_multimodal.text.encoder import encode_text
from src.phase6_multimodal.video.generator import generate_video


class MultimodalTests(unittest.TestCase):
    def test_text_encoder_and_fusion_are_deterministic(self) -> None:
        embedding_1 = encode_text("A red car", embedding_dim=32)
        embedding_2 = encode_text("A red car", embedding_dim=32)
        shared_embedding = combine_embeddings([embedding_1, embedding_2])
        alignment_score = align_modalities(embedding_1, shared_embedding)

        self.assertTrue(torch.allclose(embedding_1, embedding_2))
        self.assertEqual(shared_embedding.shape[0], 32)
        self.assertGreater(alignment_score, 0.99)

    def test_generators_create_outputs_without_existing_models(self) -> None:
        shared_embedding = encode_text("A blue bird", embedding_dim=32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            image_path = generate_image(
                shared_embedding,
                model_path=temp_dir_path / "missing_image_model.pt",
                output_path=temp_dir_path / "image.png",
            )
            audio_path = generate_audio(
                shared_embedding,
                model_path=temp_dir_path / "missing_audio_model.pt",
                output_path=temp_dir_path / "audio.wav",
                sample_rate=4000,
                duration=0.2,
            )
            video_path = generate_video(
                shared_embedding,
                model_path=temp_dir_path / "missing_video_model.pt",
                output_path=temp_dir_path / "video.gif",
                fps=6,
            )

            self.assertTrue(image_path.exists())
            self.assertTrue(audio_path.exists())
            self.assertTrue(video_path.exists())

    def test_run_multimodal_pipeline_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            config = {
                "phase6": {
                    "text_input": "A robot dancing under the stars",
                    "embedding_dim": 32,
                    "audio_sample_rate": 4000,
                    "audio_duration": 0.2,
                    "video_fps": 6,
                    "models": {
                        "image_model": str(temp_dir_path / "missing_image_model.pt"),
                        "audio_model": str(temp_dir_path / "missing_audio_model.pt"),
                        "video_model": str(temp_dir_path / "missing_video_model.pt")
                    },
                    "outputs": {
                        "image": str(temp_dir_path / "outputs" / "image.png"),
                        "audio": str(temp_dir_path / "outputs" / "audio.wav"),
                        "video": str(temp_dir_path / "outputs" / "video.gif"),
                        "report": str(temp_dir_path / "reports" / "report.json")
                    }
                }
            }

            config_path = temp_dir_path / "config_phase6.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_multimodal_pipeline(config_path)

            self.assertIn("shared_embedding", result)
            self.assertTrue((temp_dir_path / "outputs" / "image.png").exists())
            self.assertTrue((temp_dir_path / "outputs" / "audio.wav").exists())
            self.assertTrue((temp_dir_path / "outputs" / "video.gif").exists())
            self.assertTrue((temp_dir_path / "reports" / "report.json").exists())
