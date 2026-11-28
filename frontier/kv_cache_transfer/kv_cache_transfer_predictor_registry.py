from typing import TYPE_CHECKING, Optional

from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import (
    BaseKVCacheTransferPredictor,
)
from frontier.types import KVCacheTransferType
from frontier.utils.base_registry import BaseRegistry

if TYPE_CHECKING:
    from frontier.config.kv_cache_transfer_config import BaseKVCacheTransferConfig
    from frontier.kv_cache_transfer.link_profile import ResolvedKVCacheTransferProfile


class KVCacheTransferPredictorRegistry(BaseRegistry):
    """Registry for KV cache transfer predictors."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> KVCacheTransferType:
        return KVCacheTransferType.from_str(key_str)

    @classmethod
    def get(
        cls,
        predictor_type: KVCacheTransferType,
        config: "BaseKVCacheTransferConfig",
        resolved_profile: Optional["ResolvedKVCacheTransferProfile"] = None,
    ) -> BaseKVCacheTransferPredictor:
        if predictor_type not in cls._registry:
            raise ValueError(
                f"KV cache transfer predictor type {predictor_type} is not registered"
            )
        predictor_cls = cls._registry[predictor_type]
        if resolved_profile is not None:
            return predictor_cls(config, resolved_profile=resolved_profile)
        return predictor_cls(config)


KVCacheTransferPredictorRegistry.register(
    KVCacheTransferType.ANALYTICAL, AnalyticalKVCacheTransferPredictor
)
