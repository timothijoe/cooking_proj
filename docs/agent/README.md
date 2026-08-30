# Agent 开发知识库

这里保存给 agent 和维护者看的当前事实、历史与接口约束。操作者教程在
[`tutorials/`](../../tutorials/README.md)，不要把调试流水或迁移过程写进教程。

阅读顺序：

1. 先读目标领域的 `charter.md` 与 `current.md`；
2. 修改具体能力前读对应 `evolution/`；
3. 需要追溯理由时再读 `chronicles/` 和 `decisions/`；
4. 跨包调用前查阅 [`interfaces/`](interfaces/README.md)。

领域入口：[`simulation/`](simulation/README.md)、[`teleoperation/`](teleoperation/README.md)、
[`retargeting/`](retargeting/README.md)、[`real_robot/`](real_robot/README.md)。

维护规则见[文档体系设计](../design/documentation-system.md)。编年体只追加；纪传体始终
更新为最新事实；ROS 2/Wuji ROS 桥接当前不在范围内。
