# 文档体系设计

## 目标

为仿真、遥操作、重定向和真机开发建立可长期扩展的文档体系，同时兼顾历史可追溯性、当前信息准确性和实际操作教程。

## 总体结构

```text
docs/
├── agent/
│   ├── simulation/
│   ├── teleoperation/
│   ├── retargeting/
│   ├── real_robot/
│   └── interfaces/
tutorials/
├── README.md
├── environment_setup.md
├── simulation.md
├── teleoperation.md
├── retargeting.md
├── tianji_hardware.md
├── wuji_hardware.md
└── recording_and_replay.md
```

`docs/agent/` 面向 agent 和开发维护者，记录架构、当前事实、历史和接口约束。
根目录 `tutorials/` 面向实际操作者，提供可以直接照做的使用手册。两类文档不混放。

每个 `docs/agent/<domain>/` 使用相同结构：

```text
README.md
charter.md
current.md
chronicles/
evolution/
decisions/
```

## 文档职责

- `README.md`：领域导航、阅读顺序和文档索引。
- `charter.md`：长期总章程，包括边界、依赖方向、安全原则、文档职责和完成标准。它只做原则性总览，并链接相应纪传体，不重复实现细节。
- `current.md`：当前总览，包括能力、验证状态、已知限制和近期工作摘要。它只做索引和摘要，详细现状必须链接到 `evolution/` 中对应的纪传体。
- `chronicles/`：编年体，按日期记录已经发生并有证据的开发阶段。
- `evolution/`：纪传体，按功能或模块维护其最新、完整的发展说明。
- `decisions/`：对长期架构有影响的决定及其理由。

## 编年体规则

- 永久保存，只追加、不删除。
- 一篇文档覆盖一个有明确结果的开发阶段，文件名以日期开头。
- 记录当时目标、实现、验证证据、限制和后续影响。
- 后来发现旧记录错误时，不静默改写历史结论；新增勘误或后续编年记录，并链接原文。
- 不收录原始聊天、临时尝试、未经验证的推测和逐日调试流水。

## 纪传体规则

- 每项长期能力一篇，例如护手切菜、Wuji 仿真、Xbox 遥操作或手套重定向。
- 始终反映最新有效实现，允许重写和删除已经失效的旧进展。
- 正文只保留当前架构、数据流、关键约束、最新验证和已知限制。
- 历史细节通过链接指向 `chronicles/`，不在正文重复堆积。
- 功能被替代时更新原文，避免同时保留多个互相矛盾的“当前版本”。
- 它是功能级详细现状的权威来源；`charter.md` 与 `current.md` 不复制其中的大段内容。

## 架构决定规则

- 一项重大决定一个文件，使用递增编号。
- 决定不删除；失效时标记为 `superseded` 并链接替代决定。
- 普通 bug、参数微调和一次性实验不写 decision。

## Agent 文档与人类教程

- `docs/agent/interfaces/` 按模块说明 Python API、CLI、输入输出、资源依赖、依赖方向和安全语义，供 agent 修改代码前查阅。
- 根目录 `tutorials/README.md` 是面向人的教程总索引。
- 教程按任务组织，提供环境要求、可复制命令、预期现象、停止方法和常见错误。
- 明确区分 `.sh`、安装后的 CLI、`python -m` 和直接 Python 文件入口。
- 真机教程默认从帮助或 dry-run 开始，实际运动命令必须单独标识。
- 教程不包含迁移历史、内部架构争论、agent 工作流或测试台账；只保留操作者完成任务所需的信息。
- Agent 文档可以链接教程作为人工验收步骤；教程不要求读者先理解 agent 文档。

## 本轮整理范围

- 从旧项目文档中提炼仍有效的仿真演进、开发章程、遥操作、retargeting、真机接口和安全信息。
- 在 `docs/agent/` 建立上述四个领域的导航、章程/状态、编年记录和纪传文档。
- 在根目录 `tutorials/` 建立仿真、遥操作、retargeting、Tianji 真机、Wuji 真机、录制回放和环境安装教程。
- ROS 2 仅注明暂缓，不迁移其教程。
- 不迁移旧的 Superpowers 计划、原始聊天、重复状态文档和过时绝对路径。

## Git 规则

本轮允许本地提交，但未经用户明确确认不得执行 `git push`。
