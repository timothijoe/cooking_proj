# cooking_proj Repository Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库整理成一个中文、幽默、不过度包装的“厨房杂物抽屉”，配有基础忽略规则和原创横版封面图。

**Architecture:** README 负责仓库定位与内容说明，`.gitignore` 只提供技术栈无关的保守规则，`assets/kitchen-drawer.png` 作为 README 的本地相对路径封面。所有内容直接落在当前 `main` 分支，验证后推送至 `origin/main`。

**Tech Stack:** Markdown、Git ignore 规则、PNG 位图、Git/GitHub

## Global Constraints

- README 全文使用中文，简短、随意并带一点自嘲。
- 不虚构安装步骤、功能、架构、徽章或路线图。
- 图片为温暖手绘风横版厨房静物，不含文字、品牌标识或水印。
- `.gitignore` 保守且不忽略菜谱、图片或普通源码。

---

### Task 1: README 与基础忽略规则

**Files:**
- Modify: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: `assets/kitchen-drawer.png` 这一约定路径。
- Produces: GitHub 仓库首页文案，以及跨平台的基础忽略规则。

- [ ] **Step 1: 写入 README**

将 `README.md` 改为：顶部通过 `![厨房杂物抽屉](assets/kitchen-drawer.png)` 引用封面；正文包含“厨房杂物抽屉”标题、这是随手堆放不重要内容的说明、可能出现的菜谱碎片/失败记录/临时想法清单，以及“内容随缘更新、照着做饭请自行判断火候”的免责声明。

- [ ] **Step 2: 写入 `.gitignore`**

加入 `.DS_Store`、`Thumbs.db`、编辑器本地目录、`*.log`、临时文件、`.env`（但保留 `.env.example`）、`__pycache__/`、`.pytest_cache/`、`node_modules/`、`dist/`、`build/` 和 `coverage/`；不添加图片或 Markdown 通配规则。

- [ ] **Step 3: 检查文本与规则**

Run: `git diff --check && rg -n 'assets/kitchen-drawer.png|厨房杂物抽屉|自行判断火候' README.md && git check-ignore .env node_modules/demo dist/demo.log`

Expected: `git diff --check` 无输出；`rg` 找到对应内容；三个测试路径均被 `.gitignore` 匹配。

- [ ] **Step 4: 提交文本文件**

Run: `git add README.md .gitignore && git commit -m "docs: introduce the kitchen drawer"`

Expected: 新提交包含且仅包含 `README.md` 与 `.gitignore`。

### Task 2: README 封面图

**Files:**
- Create: `assets/kitchen-drawer.png`

**Interfaces:**
- Consumes: README 中的相对图片路径。
- Produces: GitHub 可直接渲染的 PNG 封面。

- [ ] **Step 1: 生成横版插画**

使用内置图像生成工具，提示词明确指定：README 横版封面；温暖厨房台面；散落的菜谱便签、木勺、香料和一只略显随意的锅；轻松、有生活感的手绘插画；自然构图与适量留白；不含文字、Logo、水印或人物。

- [ ] **Step 2: 保存并检查图片**

将最终输出复制为 `assets/kitchen-drawer.png`，再运行 `file assets/kitchen-drawer.png` 与 `identify assets/kitchen-drawer.png`（若 ImageMagick 可用）。

Expected: 文件为有效 PNG，宽度大于高度，适合横版展示。

- [ ] **Step 3: 验证 README 引用**

Run: `test -f assets/kitchen-drawer.png && rg -n '^!\[厨房杂物抽屉\]\(assets/kitchen-drawer.png\)$' README.md`

Expected: 命令退出码为 0，且打印 README 中的图片引用行。

- [ ] **Step 4: 提交封面图**

Run: `git add assets/kitchen-drawer.png && git commit -m "assets: add kitchen drawer cover"`

Expected: 新提交只包含 `assets/kitchen-drawer.png`。

### Task 3: 最终验证与 GitHub 更新

**Files:**
- Verify: `README.md`
- Verify: `.gitignore`
- Verify: `assets/kitchen-drawer.png`

**Interfaces:**
- Consumes: 前两个任务的全部交付物。
- Produces: 与本地 `main` 同步的 `origin/main`。

- [ ] **Step 1: 检查最终差异和工作树**

Run: `git diff HEAD~2..HEAD --check && git status --short --branch && git log -3 --oneline --decorate`

Expected: 无空白错误；工作树干净；最近三条提交依次包含封面、README/ignore 和设计/计划相关提交。

- [ ] **Step 2: 推送 GitHub**

Run: `git push origin main`

Expected: GitHub 接受推送，`origin/main` 更新到本地 `main`。

- [ ] **Step 3: 确认同步状态**

Run: `git status --short --branch && git rev-parse HEAD && git rev-parse origin/main`

Expected: 状态显示 `main...origin/main` 且两个提交哈希完全一致。
