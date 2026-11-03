# artifacts

这里放**需要同步给组员的评测产物和模型**，由 Git LFS 管理。

## 目录

```text
artifacts/
  latest/
    runs/                 # 当前评测结果（曲线、混淆矩阵、val 预测图、CSV、摘要）
  weights/
    helmet_v0.1.0.pt      # 当前模型权重
  manifest.json           # 版本、来源、SHA256
```

## 组员首次使用

```bash
git lfs install
git clone https://github.com/yuitsukainori3/BLCU.git
```

已经 clone 过的：

```bash
git pull
git lfs pull
```

如果 `artifacts/` 下的图片或 `.pt` 只有约 130 字节，且内容以：

```text
version https://git-lfs.github.com/spec/v1
```

开头，说明拿到的是 LFS 指针，执行 `git lfs pull` 即可。

## 当前版本

- 版本：`v0.1.0`
- 模型：`artifacts/weights/helmet_v0.1.0.pt`
- 评测产物：`artifacts/latest/runs/`
- 来源 run：`runs/detect/imgsz9602/weights/best.pt`
- 指标和实验说明见 [docs/实验记录.md](../docs/实验记录.md) 与 `artifacts/latest/runs` 中的 CSV/摘要

## 注意

- 数据集不进 `artifacts/`，仍放在 `datasets/` 或团队网盘；
- 原始 `runs/` 继续忽略，只把当前要共享的版本复制到 `artifacts/latest/`；
- 图片可能包含现场人员，仓库保持私有，不要公开转发。
