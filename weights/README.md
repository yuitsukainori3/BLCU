# weights

模型权重目录。**权重不进 Git**，只保留本说明。

使用方式：

1. 从 GitHub Releases 或团队网盘下载权重；
2. 放到本目录，例如：
   ```text
   weights/helmet_v0.1.0.pt
   ```
3. 在 `models/manifest.json` 中登记版本、来源和 SHA256；
4. 应用通过配置读取权重路径。

不要提交 `*.pt`、`*.onnx`、`*.engine`。
