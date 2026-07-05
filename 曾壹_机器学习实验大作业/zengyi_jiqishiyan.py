import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# 设置绘图风格和字体，解决中文和负号显示问题
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据导入与预处理
url = 'wine.data'
column_names = [
    'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
    'Proline'
]
data = pd.read_csv(url, header=None, names=column_names)

# 显示前5组数据
print("前5组数据：")
print(data.head())

# 定义特征和标签
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 逻辑回归模型
log_reg = LogisticRegression(max_iter=10000, multi_class='ovr')
log_reg.fit(X_train_scaled, y_train)

# 打印训练集和测试集的精度
train_accuracy = log_reg.score(X_train_scaled, y_train)
test_accuracy = log_reg.score(X_test_scaled, y_test)
print(f"\n逻辑回归模型：")
print(f"训练集精度: {train_accuracy:.4f}")
print(f"测试集精度: {test_accuracy:.4f}")

# 打印截距和权重系数
print("\n截距 (Intercepts):")
print(log_reg.intercept_)
print("\n权重系数 (Coefficients):")
coefficients = log_reg.coef_
for i, coef in enumerate(coefficients):
    print(f"类别 {i}: {coef}")

# 3. 参数C变化对权重系数的影响
C_values = np.logspace(-4, a5, num=100)
weights = []

for C in C_values:
    lr = LogisticRegression(C=C, max_iter=10000, multi_class='ovr')
    lr.fit(X_train_scaled, y_train)
    weights.append(lr.coef_.flatten())

weights = np.array(weights)

# 绘制13种特征权重系数随C变化的曲线
plt.figure(figsize=(14, 8))
for i in range(13):
    plt.plot(C_values, weights[:, i], label=f'特征 {i+1}', alpha=0.8)

plt.xscale('log')
plt.xlabel('参数 C (对数刻度)')
plt.ylabel('权重系数')
plt.title('不同参数 C 下各特征的权重系数变化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show(block=True)

# 4. 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 获取特征重要性并排序
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': importances
}).sort_values(by='重要性', ascending=False)

print("\n随机森林特征重要性排序：")
print(feature_importance_df)

# 绘制特征重要性排列图
plt.figure(figsize=(12, 8))
sns.barplot(x='重要性', y='特征', data=feature_importance_df, palette='viridis')
plt.title('红酒数据集13种特征的重要性排序（随机森林）')
plt.tight_layout()
plt.show(block=True)

# 选择阈值0.1的重要特征，不足5个则取前5
threshold = 0.1
important_features = feature_importance_df[feature_importance_df['重要性'] >= threshold]
print(f"\n阈值 {threshold} 下最重要的特征及其重要性：")
print(important_features)

if len(important_features) < 5:
    important_features = feature_importance_df.head(5)
print("\n选出的最重要的5个特征及其重要性：")
print(important_features)

# 5. 线性判别分析 (LDA) - 改用sklearn LDA类（核心修改）
# 初始化LDA，降维到2维
lda = LDA(n_components=2)
# 训练LDA并对训练集、测试集做变换
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# 打印LDA相关参数（替代手动计算的均值、散布矩阵等）
print("\nLDA 各类别均值向量：")
print(lda.means_)
print("\nLDA 类内散布矩阵的倒数与类间散布矩阵的乘积的特征值：")
print(lda.explained_variance_ratio_)

# 计算可判别比率
explained_variance_ratio = lda.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 绘制单个可判别性与累计可判别性的曲线
plt.figure(figsize=(10, 6))
plt.bar(range(1, 3), explained_variance_ratio, alpha=0.5, align='center', label='单个可判别性')
plt.step(range(1, 3), cumulative_explained_variance, where='mid', label='累计可判别性')
plt.ylabel('可判别比率')
plt.xlabel('线性判别式 (LD)')
plt.title('LDA 可判别性与累计可判别性')
plt.legend()
plt.tight_layout()
plt.show(block=True)

# 绘制LDA变换后的二维散点图
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c=c, label=f'类别 {l}', marker=m)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA 变换后的二维训练数据散点图')
plt.legend()
plt.tight_layout()
plt.show(block=True)

# 6. LDA + 逻辑回归 分类
log_reg_lda = LogisticRegression(max_iter=10000, multi_class='ovr')
log_reg_lda.fit(X_train_lda, y_train)

# 训练集预测与精度
train_pred_lda = log_reg_lda.predict(X_train_lda)
train_acc_lda = accuracy_score(y_train, train_pred_lda)
print(f"\nLDA + 逻辑回归 模型训练集精度: {train_acc_lda:.4f}")

# 测试集预测与精度
test_pred_lda = log_reg_lda.predict(X_test_lda)
test_acc_lda = accuracy_score(y_test, test_pred_lda)
print(f"LDA + 逻辑回归 模型测试集精度: {test_acc_lda:.4f}")

# 7. 绘制训练数据分类结果的背景分区图
plt.figure(figsize=(12, 8))

# 创建网格点用于背景着色
x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),  # 增加点数使背景更平滑
                     np.linspace(y_min, y_max, 200))

# 预测网格点的类别
Z = log_reg_lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 创建背景颜色映射
background_colors = np.zeros((xx.shape[0], xx.shape[1], 3))  # RGB 数组
background_colors[Z == 1] = [1, 0.8, 0.8]  # 类别1: 浅红色
background_colors[Z == 2] = [0.8, 1, 0.8]  # 类别2: 浅绿色
background_colors[Z == 3] = [0.8, 0.8, 1]  # 类别3: 浅蓝色

# 绘制背景
plt.imshow(background_colors, extent=[x_min, x_max, y_min, y_max],
           aspect='auto', origin='lower', alpha=0.6)

# 绘制决策边界线
contour = plt.contour(xx, yy, Z, levels=[1.5, 2.5], colors='black',
                      linewidths=2, linestyles='-')
plt.clabel(contour, inline=True, fontsize=10, fmt='决策边界 %d')

# 绘制原始数据点（在背景之上）
colors_point = ['red', 'green', 'blue']
markers = ['s', 'x', 'o']  # s:方形, x:叉号, o:圆形
for l, c, m in zip(np.unique(y_train), colors_point, markers):
    # 根据标记类型调整样式
    if m == 'x':
        plt.scatter(X_train_lda[y_train == l, 0],
                    X_train_lda[y_train == l, 1],
                    c=c, label=f'类别 {l} (真实)', marker=m, s=60)
    else:
        plt.scatter(X_train_lda[y_train == l, 0],
                    X_train_lda[y_train == l, 1],
                    c=c, label=f'类别 {l} (真实)', marker=m, s=60,
                    edgecolors='k', linewidth=0.8)

plt.xlabel('LD1', fontsize=12)
plt.ylabel('LD2', fontsize=12)
plt.title('LDA 变换后的训练数据分类结果（背景分区图）', fontsize=14, fontweight='bold')
plt.legend(title='图例', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)

# 8. 绘制测试数据分类结果的背景分区图
plt.figure(figsize=(12, 8))

# 使用相同的网格范围
xx_test, yy_test = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))

# 预测测试集网格点的类别
Z_test = log_reg_lda.predict(np.c_[xx_test.ravel(), yy_test.ravel()])
Z_test = Z_test.reshape(xx_test.shape)

# 创建背景颜色映射
background_colors_test = np.zeros((xx_test.shape[0], xx_test.shape[1], 3))
background_colors_test[Z_test == 1] = [1, 0.8, 0.8]  # 类别1: 浅红色
background_colors_test[Z_test == 2] = [0.8, 1, 0.8]  # 类别2: 浅绿色
background_colors_test[Z_test == 3] = [0.8, 0.8, 1]  # 类别3: 浅蓝色

# 绘制背景
plt.imshow(background_colors_test, extent=[x_min, x_max, y_min, y_max],
           aspect='auto', origin='lower', alpha=0.6)

# 绘制决策边界线
contour_test = plt.contour(xx_test, yy_test, Z_test, levels=[1.5, 2.5],
                           colors='black', linewidths=2, linestyles='-')
plt.clabel(contour_test, inline=True, fontsize=10, fmt='决策边界 %d')

# 绘制测试数据点
for l, c, m in zip(np.unique(y_test), colors_point, markers):
    if m == 'x':
        plt.scatter(X_test_lda[y_test == l, 0],
                    X_test_lda[y_test == l, 1],
                    c=c, label=f'类别 {l} (真实)', marker=m, s=60)
    else:
        plt.scatter(X_test_lda[y_test == l, 0],
                    X_test_lda[y_test == l, 1],
                    c=c, label=f'类别 {l} (真实)', marker=m, s=60,
                    edgecolors='k', linewidth=0.8)

plt.xlabel('LD1', fontsize=12)
plt.ylabel('LD2', fontsize=12)
plt.title('LDA 变换后的测试数据分类结果（背景分区图）', fontsize=14, fontweight='bold')
plt.legend(title='图例', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)