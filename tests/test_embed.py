import torch

from eo_data_embedding.embed import ViTEmbedder


def test_vit_embedder_shape_and_finite():
    # pretrained=False -> random weights, no network download
    emb = ViTEmbedder(backbone="vit_tiny_patch16_224", in_chans=3, pretrained=False, device="cpu")
    out = emb.encode(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, emb.embed_dim)
    assert torch.isfinite(out).all()
