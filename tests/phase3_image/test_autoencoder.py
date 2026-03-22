import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.common.utils import save_training_history
from src.phase3_image.autoencoder import (
    initialize_autoencoder,
    load_autoencoder,
    reconstruction_loss,
    save_autoencoder,
)
from src.phase3_image.dataset import create_image_data_loader, flatten_images, split_image_dataset
from src.phase3_image.pipeline import save_reconstruction_samples
from src.phase3_image.train import (
    initialize_autoencoder_optimizer,
    run_autoencoder_training,
    train_autoencoder,
    train_autoencoder_epoch,
    validate_autoencoder,
)


class AutoencoderTests(unittest.TestCase):
    def setUp(self) -> None:
        images = torch.rand(12, 1, 28, 28)
        labels = torch.zeros(12, dtype=torch.long)
        self.dataset = TensorDataset(images, labels)

    def test_autoencoder_encode_decode_and_forward_shapes(self) -> None:
        model = initialize_autoencoder(input_dim=784, latent_dim=16)
        image_batch, _ = self.dataset[:4]
        flattened_images = flatten_images(image_batch)

        latent_vector = model.encode(flattened_images)
        reconstructed_images = model.decode(latent_vector)
        forward_output = model(flattened_images)

        self.assertEqual(latent_vector.shape, (4, 16))
        self.assertEqual(reconstructed_images.shape, (4, 784))
        self.assertEqual(forward_output.shape, (4, 784))

    def test_reconstruction_loss_returns_scalar(self) -> None:
        original = torch.rand(2, 784)
        reconstructed = torch.rand(2, 784)
        loss = reconstruction_loss(original, reconstructed)
        self.assertEqual(loss.ndim, 0)

    def test_autoencoder_training_loop_and_artifacts(self) -> None:
        train_dataset, val_dataset, test_dataset = split_image_dataset(self.dataset, 0.6, 0.2, 0.2)
        train_loader = create_image_data_loader(train_dataset, batch_size=2, shuffle=True)
        val_loader = create_image_data_loader(val_dataset, batch_size=2, shuffle=False)
        test_loader = create_image_data_loader(test_dataset, batch_size=2, shuffle=False)

        model = initialize_autoencoder(input_dim=784, latent_dim=16)
        optimizer = initialize_autoencoder_optimizer(model, learning_rate=0.001)

        batch_images, _ = next(iter(train_loader))
        batch_loss = train_autoencoder(model, batch_images, optimizer)
        train_loss = train_autoencoder_epoch(model, train_loader, optimizer)
        val_loss = validate_autoencoder(model, val_loader)
        trained_model, training_history = run_autoencoder_training(
            model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=2,
        )

        self.assertGreaterEqual(batch_loss, 0.0)
        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertEqual(len(training_history), 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_path = temp_dir_path / "autoencoder.pt"
            history_path = temp_dir_path / "training_history.json"
            recon_path = temp_dir_path / "recon.png"

            save_autoencoder(trained_model, model_path)
            loaded_model = load_autoencoder(model_path)
            save_training_history(training_history, history_path)

            sample_batch, _ = next(iter(test_loader))
            save_reconstruction_samples(loaded_model, sample_batch, recon_path)

            self.assertTrue(model_path.exists())
            self.assertTrue(history_path.exists())
            self.assertTrue(recon_path.exists())

            history = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(len(history), 2)
