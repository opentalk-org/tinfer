from typing import Optional
from ..utils import exists, rand_bool
import torch
import torch.nn as nn
from einops import reduce
from einops.layers.torch import Rearrange
from torch import Tensor
from .attention import TransformerBlock
from .embeddings import FixedEmbedding, TimePositionalEmbedding


class Transformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )

        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features
        )


    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)

        return mapping

    def run(self, x, time, embedding, features):

        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], axis=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)

        for block in self.blocks:
            x = x + mapping
            x = block(x)

        x = x.mean(axis=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)

        return x

    def forward(self, x: Tensor,
                time: Tensor,
                embedding_mask_proba: float = 0.0,
                embedding: Optional[Tensor] = None,
                features: Optional[Tensor] = None,
               embedding_scale: float = 1.0) -> Tensor:

        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)

        return x
