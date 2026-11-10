## **一、当前版本和同步事实**

2026-11-10 只读执行 `git ls-remote origin refs/heads/main`，远端 `main` 为 `54345ca92dd05c8a393b7e4b4614b1053567fc14`。仓库为 [luminoso909/MiniGen](https://github.com/luminoso909/MiniGen)，本地分支 `main`。审查起点 `77b750685f850f1b6789995fc3dd5a1a74857b87` 比该远端领先 10 个提交；本轮完成后增加一次本地提交。该远端快照可能以后变化，因此用户 Push 前再查询。

本轮没有 Push，没有 reset、force push 或修改 Git 全局配置。用户此前未提交的 Notebook、week1QA 删除、docs/QA 和 VAE review 文件夹维持原状，不纳入提交。数据、权重、缓存、检查点和整批生成图片仍由 `.gitignore` 排除。

## **二、用户在 Mac 主动同步**

以下由用户批准后在 Mac 执行。Push 会包含本次提交及远端缺失的已有提交，不会提交工作区中的未提交文件。

```bash
cd /Users/lucius/projects/personal/MiniGen
git status --short
git log -1 --format='%H%n%s'
git ls-remote origin refs/heads/main
git push origin main
git ls-remote origin refs/heads/main
```

最后一次远端 SHA 应与要交付的本地 SHA 一致。若非 fast-forward 拒绝，先检查远端新增提交，不要 force push。若后来继续做了别的提交，请从本轮最终回复记录完整 SHA，不用“最新 HEAD”替代已经选定的实验版本。

未提交改动不影响同步已提交源码；不要为得到 clean 状态删除用户笔记。云端克隆的是提交内容，不包括这些未提交学习修改。另行同步资产时也不复制 Mac `.venv` 或凭据。

## **三、云端固定源码与恢复边界**

使用 [完整手册](../../../docs/cloud/MiniGen_云端训练与评测完整操作手册.md) 的 `git clone`、交互填写完整 SHA、`git checkout --detach` 和 HEAD 校验。不要在本地代为运行云端命令。

固定提交、`uv.lock`、配置/Manifest/组件/缓存哈希一起构成迁移依据。采样可以读取有明确兼容规则的旧检查点；精确训练恢复还要求源码、Torch、设备类型和完整配方一致。本轮 VQA 问句修复改变全包源码身份，因此应在本轮固定源码上新建 Pilot，不绕过旧检查点的严格恢复检查。换 GPU 型号不保证逐位恢复。

`.gitignore` 覆盖 `/data/*`、`/artifacts/*`、`*.pt/*.pth/*.ckpt/*.safetensors/*.npy/*.npz/*.parquet` 和常见凭据。本轮暂存只按白名单加入代码、配置、报告与导航；不会依赖忽略规则替代实际暂存检查。模型 `.bin` 存在 `artifacts/models/` 内，由目录规则排除，不要移到源码目录。

若不希望 Push，可按手册使用 Git bundle 迁移同一提交。资产仍走独立 rsync/锁驱动下载与校验；GitHub 上有源码不等于云端已有数据或权重。正式训练前核对文件及独立回读，不重复构建已批准的 P2 缓存。
