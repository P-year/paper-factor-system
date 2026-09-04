"""
pipelines/ - 业务 pipeline 注册目录

每个子目录 = 一个 pipeline：
  pipelines/<name>/
    config.yaml      声明：prompts / hard_rules / max_iter / interrupt_before / terminal_nodes
    state.py         TypedDict：继承 BasePipelineState，加自己的字段
    nodes/           node 函数体（每个一个文件）
    auto_approve.py  scheduler 模式的自动审批回调（可选）
    __init__.py      @register_pipeline 注册入口
"""