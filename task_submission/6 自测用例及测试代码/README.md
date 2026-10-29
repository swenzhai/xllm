<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# 自测用例与测试代码

`官方任务包/` 保存发布包全部12个文件的逐字节副本：35个公开case、10个性能/内存适用项、coverage_requirements、独立CPU golden、数据生成器、公开验证器、benchmark及性能检查器。原始SHA256SUMS中的任务书名称/哈希异常原样保留，详见根目录步骤说明；交付副本的新哈希见 `../7 待验收工程/源码文件清单.json`。

`扩展测试/tests/` 是被测固定提交的5个测试源码副本，`扩展测试/common/` 保留host映射测试依赖的共享tiling头；目录层级与原工程一致。`acceptance.py` 覆盖工程契约、非法输入、准备接口、stream、生命周期、连续追加、workspace及分配器内存；另有队列压力、缓存边界和纯CPU地址映射检查。副本没有改变算法或测试判据。

无需NPU即可检查交付结构和公开case清单（Python3和C++17编译器）：

```bash
export SUBMISSION=/absolute/path/to/task_submission
export TASK_PACKAGE="$SUBMISSION/6 自测用例及测试代码/官方任务包"
export EXT_TESTS="$SUBMISSION/6 自测用例及测试代码/扩展测试/tests"
python3 "$EXT_TESTS/acceptance.py" --package-dir "$TASK_PACKAGE" --list
python3 "$EXT_TESTS/check_task_mapping.py"
```

NPU复现需先按 `../7 待验收工程/源码与工程索引.md` 获取固定Git工程，并按 `../1 自验证步骤说明.md` 完成构建、私有安装和环境设置。`T5QKV_SOLUTION` 必须指向该工程的 `solution.py`；测试目录本身不包含产品动态库。以下命令在上述准备完成后执行，完整预热/计时/检查器协议见步骤说明：

```bash
(cd "$TASK_PACKAGE" && python3 verify.py --solution-file "$T5QKV_SOLUTION" \
  --report "$T5QKV_REPORTS/public_accuracy.json")
python3 "$EXT_TESTS/acceptance.py" --package-dir "$TASK_PACKAGE" \
  --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/extended.json"
python3 "$EXT_TESTS/queue_stress.py" --package-dir "$TASK_PACKAGE" \
  --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/queue_stress.json"
python3 "$EXT_TESTS/cache_boundary.py" --solution-file "$T5QKV_SOLUTION" \
  --report "$T5QKV_REPORTS/cache_boundary.json"
```

35个官方case不能由扩展统计替代，未标记performance的25个case仍需精度验证。纯CPU映射检查和 `--list` 不证明NPU正确性或性能。本次已有实测记录见原始证据ZIP；此目录提供源码与复现入口，不生成新的硬件验证声明。
