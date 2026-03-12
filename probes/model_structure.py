from __future__ import annotations


def get_text_model(model):
    candidates = [
        lambda m: getattr(getattr(m, "model", None), "language_model", None),
        lambda m: getattr(m, "language_model", None),
        lambda m: getattr(m, "model", None),
        lambda m: getattr(m, "text_model", None),
        lambda m: getattr(getattr(m, "model", None), "text_model", None),
    ]
    for candidate in candidates:
        text_model = candidate(model)
        if text_model is not None and hasattr(text_model, "layers"):
            return text_model
    raise AttributeError(f"Could not locate transformer text model for {type(model).__name__}")


def get_transformer_layers(model):
    text_model = get_text_model(model)
    return text_model.layers


def get_transformer_layer(model, layer_idx: int):
    return get_transformer_layers(model)[layer_idx]


def get_embed_tokens_module(model):
    text_model = get_text_model(model)
    if hasattr(text_model, "embed_tokens"):
        return text_model.embed_tokens
    raise AttributeError(f"Could not locate embed_tokens for {type(model).__name__}")
