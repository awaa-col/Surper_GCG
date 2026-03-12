# Family Structure

这一组实验专门回答两件事：

1. `late safety families` 是否稳定存在，而不是当前 Scope 配置下的偶然切片。
2. `refusal / risk / empathy / resource / unsafe_exec` 这些 family 到底各自起什么作用、彼此如何依赖。

新实验：

- `exp_32_family_stability.py`
  family 稳定性与跨切分一致性。
- `exp_33_family_causal_sweep.py`
  单簇因果干预。
- `exp_34_family_dependency_matrix.py`
  family 间依赖矩阵。

历史相关实验：

- `../exp_16_safe_response_dictionary.py`
- `../exp_17_gemma_scope_feature_probe.py`
- `../exp_19_l17_l23_late_impact.py`
- `../exp_20_prefill_soft_apology_probe.py`
- `../exp_26_vector_interaction_map.py`
- `../exp_27_vector_effect_atlas.py`
- `../exp_28_detect_family_causal.py`
