## **一、学习源码与运行方式**

这里保存 S1–S3 的小模型、课程算法、旧数据工具和历史实验包装，供重读和复验。正式安装包只包含 `src/minigen/`；生产代码不会自动查找这里的模块。旧 Notebook 的导入保持原样，需要时显式启用学习路径：

```bash
PYTHONPATH=docs/learning/src uv run --locked --extra notebook jupyter lab
```

上面的命令从仓库根目录启动。仓库内的 [Notebook 执行工具](../../../scripts/execute_learning.py) 和个人复验脚本会为各自进程显式启用同一学习路径。删除的旧 MNIST 下载、合成图标或 128px 缓存需要复现实验时按对应 [历史报告](../../../reports/README.md) 重建；保留的实验结果不依赖重跑 Notebook。
