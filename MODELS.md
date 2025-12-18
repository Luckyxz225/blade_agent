# 模型文件下载说明

本项目依赖以下预训练模型文件，由于文件过大（>100MB），未包含在 Git 仓库中。

## 所需模型文件

请将以下文件放置在 `all_file_path/` 目录下：

| 文件名 | 大小 | 说明 |
|-------|------|------|
| `best_model_in_testdataset.pth` | 175 MB | 性能预测最优模型 |
| `pre_guide_15200.pth` | 24 MB | 扩散模型 |
| `prior_parameters_guide_15200.pth` | 21 MB | 先验参数生成模型 |
| `parameter_geometry_5200.npy` | 630 KB | 几何参数数据集 |
| `label_5200.npy` | 90 KB | 标签数据集 |

## 下载方式

### 方式 1：网盘下载（推荐）

- **百度网盘**：[待添加链接]
- **提取码**：[待添加]

### 方式 2：联系作者获取

如果您需要这些模型文件，请通过以下方式联系：
- Email: [your_email@example.com]
- GitHub Issue: [项目 Issues 页面]

## 快速设置

下载后，将文件放置到正确位置：

```bash
# 解压后应该是这样的目录结构
Blade_agent/
├── all_file_path/
│   ├── best_model_in_testdataset.pth
│   ├── pre_guide_15200.pth
│   ├── prior_parameters_guide_15200.pth
│   ├── parameter_geometry_5200.npy
│   └── label_5200.npy
├── generative_design/
└── ...
```

## 验证

运行以下命令验证文件是否就绪：

```bash
python -c "import os; files=['best_model_in_testdataset.pth','pre_guide_15200.pth','prior_parameters_guide_15200.pth','parameter_geometry_5200.npy','label_5200.npy']; print('✅ 所有文件就绪' if all(os.path.exists(f'all_file_path/{f}') for f in files) else '❌ 缺少模型文件')"
```

