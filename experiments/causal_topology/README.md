# Causal Topology

这一组实验专门追问：

1. `detect-like -> exec -> late families` 谁先动、谁驱动谁。
2. generation 中途切换时，轨迹的断点和可逆点在哪里。

新实验：

- `exp_35_detect_exec_late_trace.py`
  多 schedule 对照的逐 token 时序因果图。
- `exp_36_boundary_state_profile.py`
  统一比较 `soft_apology / resource_redirect / disclaimer_danger / direct_unsafe / dan` 的中间态剖面。
- `exp_37_boundary_lineage.py`
  比较 `soft_apology / disclaimer_danger / direct_unsafe` 是否是同一边界谱系上的不同强度点。

历史相关实验：

- `../exp_15_detect_survey.py`
- `../exp_19_l17_l23_late_impact.py`
- `../exp_21_dangerous_reply_token_probe.py`
- `../exp_25_generation_step_trace.py`
- `../exp_29_pure_detect_disentangle.py`
- `../exp_30_detect_signed_sweep.py`
- `../exp_31_generation_step_detect_schedule.py`
