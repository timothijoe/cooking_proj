# 开发环境配置设计

## 目标与范围

为 `cooking_proj` 建立可复现的 Python 3.12 开发环境，优先验证 MuJoCo 仿真与离线数据处理。
同时建立独立的真机环境，但只验证解释器、导入和命令帮助；不连接、使能或移动 Tianji/Wuji
设备。

已确认迁移所需的本地资源已位于 `local/`：MuJoCo 资产、MarvinCCS 模型、Tianji Python SDK、
厂商支持文件及录制数据。它们保持 Git 忽略，不复制、不修改。

## 方案比较

1. 只创建 `.venv`（推荐）：先覆盖后续日常开发所需的仿真、数据处理和测试；风险最低，
   且可用 headless 命令验证。
2. 同时创建 `.venv` 与 `.venv-hardware`：在方案一基础上预装厂商 Python 依赖，但仍不触碰
   硬件。用户已选择此方案。
3. 连接硬件做端到端验证：超出本次范围，必须有设备旁操作者、急停与明确的单独授权。

## 实施设计

- 在仓库根目录运行现有 `scripts/setup/create_env.sh`，生成 `.venv`；该脚本按
  `pyproject.toml` 安装仿真、数据、开发依赖和五个本地 editable Python 包。
- 在同一根目录运行 `scripts/setup/create_hardware_env.sh`，生成 `.venv-hardware`；该脚本把
  `local/` 写入该环境的 `.pth`，供厂商 SDK 导入，但不运行硬件控制逻辑。
- 不创建或提交 `configs/local.toml`：当前代码以 `COOKING_LOCAL_ROOT` 和标准 `local/` 结构
  解析资源，模板中的控制器 IP 不应在未确认设备网络前固化为本机配置。

## 验证与安全

- 确认两套解释器均为 Python 3.12，且 `twin-sim --help`、`tianji-robot --help` 可执行。
- 在 `.venv` 运行仓库测试和两个 headless 仿真入口；若资源或显示依赖不满足，记录准确缺口，
  不通过猜测性修改掩盖问题。
- 在 `.venv-hardware` 仅运行 `arm_command_diagnostic --help` 与安全的模块导入检查；绝不使用
  `--execute`，也不调用任何网络连接或设备初始化。
- 验证结束后报告通过项、失败项与尚待实体设备复验的项目。
