# npy-contract 中文说明

在加载数组之前，用 JSON 约定检查一批 NPY 文件：dtype、shape、存储顺序、文件长度、
单文件与合计声明内存。适用于特征、标签、索引等文件的交付检查，避免行数不一致、
字节序错误或文件截断到下游任务才暴露。运行时只需 Python 3.11+ 标准库。

```sh
python -m pip install .
python examples/make_demo.py demo
python -m npy_contract demo/contract.json --root demo/good
python -m npy_contract demo/contract.json --root demo/mismatch
```

先按[英文完整说明](README.md)创建虚拟环境。Windows 无需激活，命令中的 `python`
可替换为 `.venv/Scripts/python.exe`。第一个检查退出 0；第二个因标签与特征的样本数
不同退出 1。示例生成器只写入新目录，不覆盖已有目录。

shape 中整数表示精确维度，`null` 为任意维度，字符串如 `samples` 在整批文件中绑定为
相同数值。dtype 要与磁盘描述符完全一致，order 为 `C`、`F` 或 `either`。报告只含相对
文件名及元数据，不含数组值。任何失败都不会被标为通过；约定无效或根目录无效退出 2。

支持 NPY 1.0/2.0/3.0 的明确列出的标量数值类型；对象/pickle、结构体、字符串、日期、
扩展浮点、NPZ 均明确拒绝。只检查元数据和原始数据字节数，不验证数值、校验和或模型。
等长数据损坏可能通过。Header 上限 10,000 字节，rank 上限 32，数组数上限 1,024。
只读有界 header，不分配数组内存，不联网。请保持输入文件稳定；符号链接与 Windows
重解析点会被拒绝，不保证抵御检查期间的恶意文件替换。

开发验证命令、完整 schema、路径限制与测试流程见英文说明。NumPy 只用于开发阶段的
真实文件对照测试，不是运行依赖。卸载：`python -m pip uninstall npy-contract`，再删除
自己生成的 demo 和可选虚拟环境。MIT 许可证，安全报告方式见 SECURITY.md。
