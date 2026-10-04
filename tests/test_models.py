import pytest
import torch

from genai_a1.models import ConvAutoencoder, CorruptionClassifier, SoftMixture


def test_autoencoder_dropout_is_training_only():
    model = ConvAutoencoder(base=4, bottleneck=8, dropout=0.5)
    image = torch.rand(1, 3, 128, 128)
    model.train()
    assert not torch.equal(model(image), model(image))
    model.eval()
    assert torch.equal(model(image), model(image))


def test_soft_mixture_temperature_controls_routing_sharpness():
    gate = CorruptionClassifier(base=4, dropout=0.0).eval()
    experts = [ConvAutoencoder(base=4, bottleneck=8, dropout=0.0).eval() for _ in range(3)]
    sharp = SoftMixture(gate, experts, temperature=0.5).eval()
    smooth = SoftMixture(gate, experts, temperature=2.0).eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        _, sharp_weights = sharp(image)
        _, smooth_weights = smooth(image)
    entropy = lambda weights: -(weights * weights.clamp_min(1e-12).log()).sum(1)
    assert torch.allclose(sharp_weights.sum(1), torch.ones(1))
    assert entropy(sharp_weights).item() < entropy(smooth_weights).item()


def test_soft_mixture_rejects_nonpositive_temperature():
    gate = CorruptionClassifier(base=4)
    experts = [ConvAutoencoder(base=4, bottleneck=8) for _ in range(3)]
    with pytest.raises(ValueError, match="positive"):
        SoftMixture(gate, experts, temperature=0.0)