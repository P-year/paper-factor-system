"""
harness/ - 跨 pipeline 通用框架

职责：
- pipeline.py       PipelineConfig dataclass + yaml 加载
- registry.py       @register_pipeline + load_all + get
- state_base.py     BasePipelineState TypedDict
- graph_builder.py  build_graph(pipeline) → (compiled, checkpointer)
- plan_node.py      make_plan_node(pipeline) 工厂
- hard_rules.py     StateView + 命名谓词 DSL 求值器
- llm_client.py     LLM 调用封装（DeepSeek 主，GLM 备用）
- observability.py  langfuse 观测装饰器
- checkpoints.py    JSON 持久化（sessions / runs / audit）
- paths.py          共享项目路径
- cli.py            argparse 入口（run / list-pipelines / show）
"""