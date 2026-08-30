# Documentation Organizer Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一份可用于任意项目的参数化中文文档整理 Prompt。

**Architecture:** 单一 Markdown 模板集中参数、文档结构、更新规则、执行流程和验收标准。模板不依赖 cooking_proj 的具体领域或路径。

**Tech Stack:** Markdown、Git。

## Global Constraints

- 编年体永久保存、只追加；纪传体保持最新并清理失效内容。
- Agent 文档与人类教程分离。
- 大资源和虚拟环境不进入 Git。
- 不虚构验证，不把旧项目证据冒充当前证据。
- 未经单独明确授权不得执行 `git push`、连接真机或实际运动。

---

### Task 1: 通用 Prompt

**Files:**
- Create: `PROMPTS/documentation_organizer.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 用户填写的项目路径、参考路径、领域列表、本地资源目录和 Git 权限。
- Produces: 一份可以直接复制给 Agent 的完整任务说明。

- [x] **Step 1: 编写参数化模板**

写明参数、目标目录、编年体/纪传体规则、轻量化、安全、验证和交付要求。

- [x] **Step 2: 增加仓库入口**

在 `README.md` 链接该模板，说明它可用于其他项目。

- [x] **Step 3: 验证模板**

运行 Markdown 本地链接检查、占位符字段检查、`git diff --check`，预期均返回 0。

- [x] **Step 4: 本地提交**

```bash
git add PROMPTS/documentation_organizer.md README.md docs/design/reusable-documentation-prompt.md docs/superpowers/plans/2026-08-30-documentation-organizer-prompt.md
git commit -m "docs: add reusable documentation organizer prompt"
```
