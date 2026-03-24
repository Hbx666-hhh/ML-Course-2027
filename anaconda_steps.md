# 在Anaconda Prompt上运行实验二的步骤

## 步骤1：打开Anaconda Prompt
1. 在Windows开始菜单中搜索"Anaconda Prompt"
2. 点击打开Anaconda Prompt应用程序

## 步骤2：激活已创建的ml_env环境
```bash
# 激活ml_env环境
conda activate ml_env
```

## 步骤3：安装所需的依赖包
```bash
# 安装numpy、matplotlib和scikit-learn
pip install numpy matplotlib scikit-learn
```

## 步骤4：导航到实验目录
```bash
# 进入实验二目录
cd C:\Users\Lenovo\Desktop\z-实验二
```

## 步骤5：运行实验脚本
```bash
# 运行实验脚本
python experiment2.py
```

## 步骤6：查看实验结果
- 实验完成后，会在当前目录生成三个PNG文件：
  - probability_modeling.png：概率建模数据可视化
  - gradient_descent_loss.png：梯度下降损失函数
  - gradient_descent_fit.png：梯度下降拟合结果
- 同时在终端中会显示实验的详细输出，包括KNN分类的准确率和梯度下降优化后的参数

## 注意事项
- 如果遇到权限问题，请以管理员身份运行Anaconda Prompt
- 如果安装依赖包时遇到网络问题，可以尝试使用国内镜像源
  ```bash
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy matplotlib scikit-learn
  ```
- 确保Anaconda已正确安装并添加到系统环境变量中
