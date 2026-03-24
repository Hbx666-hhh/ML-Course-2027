import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 概率建模
def probability_modeling():
    print("=== 概率建模 ===")
    # 生成二分类数据集
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # 可视化数据
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1')
    plt.title('Generated Dataset for Probability Modeling')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('probability_modeling.png')
    print("概率建模数据可视化已保存为 probability_modeling.png")
    
    return X, y

# 2. KNN分类
def knn_classification(X, y, k=5):
    print("\n=== KNN分类 ===")
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 计算测试集中每个样本的KNN预测
    y_pred = []
    for test_point in X_test:
        # 计算与所有训练样本的距离
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        # 选择K个最近邻
        k_indices = np.argsort(distances)[:k]
        # 多数投票
        k_nearest_labels = y_train[k_indices]
        prediction = np.bincount(k_nearest_labels).argmax()
        y_pred.append(prediction)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN分类准确率 (k={k}): {accuracy:.4f}")
    
    return X_train, X_test, y_train, y_test, y_pred

# 3. 梯度下降优化
def gradient_descent():
    print("\n=== 梯度下降优化 ===")
    # 生成线性回归数据
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # 添加偏置项
    X_b = np.c_[np.ones((100, 1)), X]
    
    # 初始化参数
    theta = np.random.randn(2, 1)
    
    # 梯度下降参数
    learning_rate = 0.1
    n_iterations = 1000
    m = 100
    
    # 存储损失值
    loss_history = []
    
    # 执行梯度下降
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        
        # 计算损失
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        loss_history.append(loss)
    
    # 可视化损失函数
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations), loss_history)
    plt.title('Loss Function During Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.savefig('gradient_descent_loss.png')
    print("梯度下降损失函数可视化已保存为 gradient_descent_loss.png")
    
    # 可视化拟合结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, X_b.dot(theta), color='red', label='Fitted Line')
    plt.title('Linear Regression with Gradient Descent')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('gradient_descent_fit.png')
    print("梯度下降拟合结果可视化已保存为 gradient_descent_fit.png")
    
    print(f"梯度下降优化后的参数: theta0={theta[0][0]:.4f}, theta1={theta[1][0]:.4f}")
    
    return theta

# 主函数
def main():
    print("实验二: 概率建模、KNN分类与梯度下降优化")
    print("=" * 50)
    
    # 1. 概率建模
    X, y = probability_modeling()
    
    # 2. KNN分类
    X_train, X_test, y_train, y_test, y_pred = knn_classification(X, y)
    
    # 3. 梯度下降优化
    theta = gradient_descent()
    
    print("\n实验完成！")
    print("生成的文件:")
    print("- probability_modeling.png: 概率建模数据可视化")
    print("- gradient_descent_loss.png: 梯度下降损失函数")
    print("- gradient_descent_fit.png: 梯度下降拟合结果")

if __name__ == "__main__":
    main()
