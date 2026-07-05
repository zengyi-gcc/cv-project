import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DeepLearningProject:
    """深度学习大作业主类"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
        self.history = {}

    def load_data(self):
        """数据加载与预处理"""
        print("=== 步骤1: 数据加载与预处理 ===")

        # 数据增强策略说明
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转增加多样性
            transforms.RandomCrop(32, padding=4),  # 随机裁剪防止过拟合
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # 加载CIFAR-10数据集
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )

        # 创建DataLoader - 参数设置说明
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=64,  # 平衡内存效率与训练稳定性
            shuffle=True,  # 打乱数据避免顺序偏差
            num_workers=2  # 加速数据加载
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,  # 测试集不需要打乱
            num_workers=2
        )

        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        print(f"数据加载完成! 设备: {self.device}")

        return True

    def visualize_samples(self):
        """训练集数据可视化"""
        print("\n=== 步骤2: 数据可视化 ===")

        # 获取一个batch的数据
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)

        # 创建可视化图像
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            ax = axes[i // 5, i % 5]
            # 反标准化显示图像
            image = images[i].numpy().transpose((1, 2, 0))
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title(f'标签: {self.class_names[labels[i]]}')
            ax.axis('off')

        plt.suptitle('CIFAR-10训练集样本可视化', fontsize=16)
        plt.tight_layout()
        plt.savefig('training_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("样本可视化完成! 图像已保存为 'training_samples.png'")

    def build_model(self, model_name='resnet18'):
        """模型构建与修改最后一层"""
        print(f"\n=== 步骤3: 构建{model_name}模型 ===")

        if model_name == 'resnet18':
            # 加载预训练ResNet18模型
            model = models.resnet18(weights='IMAGENET1K_V1')
            print("✓ 使用ResNet18预训练权重")

            # 修改最后一层全连接层
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 10)  # CIFAR-10有10个类别
            print(f"✓ 修改全连接层: {in_features} -> 10")

        elif model_name == 'mobilenet_v2':
            # 加载预训练MobileNetV2模型
            model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            print("✓ 使用MobileNetV2预训练权重")

            # 修改分类器最后一层
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 10)
            print(f"✓ 修改分类器层: {in_features} -> 10")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        return model

    def train(self, model, model_name='resnet18'):
        """模型训练过程"""
        print(f"\n=== 步骤4: 开始训练{model_name}模型 ===")

        model = model.to(self.device)

        # 超参数设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 40

        print("超参数设置:")
        print(f"- 优化器: Adam")
        print(f"- 学习率: 0.001")
        print(f"- 损失函数: CrossEntropyLoss")
        print(f"- 训练轮数: {num_epochs}")

        # 记录训练过程
        history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'epoch_time': []
        }

        best_acc = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # 训练阶段
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(self.train_loader)

            # 测试阶段
            test_loss, test_correct, test_total = 0.0, 0, 0
            model.eval()

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_acc = 100 * test_correct / test_total
            avg_test_loss = test_loss / len(self.test_loader)

            epoch_time = time.time() - epoch_start

            # 记录历史数据
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(avg_test_loss)
            history['test_acc'].append(test_acc)
            history['epoch_time'].append(epoch_time)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'best_{model_name}.pth')
                print(f"✓ 保存最佳模型，准确率: {best_acc:.2f}%")

            print(f'Epoch [{epoch + 1}/{num_epochs}] - '
                  f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}% | '
                  f'测试损失: {avg_test_loss:.4f}, 测试准确率: {test_acc:.2f}% | '
                  f'时间: {epoch_time:.1f}s')

        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.1f}s, 最佳准确率: {best_acc:.2f}%")

        return history, best_acc

    def visualize_training(self, history, model_name):
        """训练过程可视化"""
        print(f"\n=== 步骤5: 生成{model_name}训练可视化 ===")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(history['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(history['test_loss'], 'r-', label='测试损失', linewidth=2)
        ax1.set_xlabel('训练轮次 (Epoch)')
        ax1.set_ylabel('损失值 (Loss)')
        ax1.set_title(f'{model_name} - 训练与测试损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(history['test_acc'], 'r-', label='测试准确率', linewidth=2)
        ax2.set_xlabel('训练轮次 (Epoch)')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title(f'{model_name} - 训练与测试准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'training_curves_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"训练曲线已保存为 'training_curves_{model_name}.png'")

    def compare_models(self, results):
        """模型对比分析"""
        print("\n=== 步骤6: 模型对比分析 ===")

        print("\n" + "=" * 60)
        print("模型对比结果表")
        print("=" * 60)
        print(f"{'模型名称':<15} {'测试准确率':<12} {'参数量':<15} {'训练时间(s)':<12}")
        print("-" * 60)

        for model_name, info in results.items():
            print(f"{model_name:<15} {info['accuracy']:<12.2f}% {info['params']:<15,} {info['time']:<12.1f}")

        print("=" * 60)

        # 简单分析
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n分析结论:")
        print(f"- 最佳性能模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.2f}%)")
        print(f"- 建议: 可尝试增加训练轮数或调整学习率以提升性能")

    def run_complete_experiment(self):
        """运行完整实验流程"""
        print("深度学习大作业 - 完整实验开始")
        print("=" * 50)

        # 1. 数据准备
        if not self.load_data():
            return

        # 2. 数据可视化
        self.visualize_samples()

        # 3. 模型训练与比较
        models_to_train = ['resnet18', 'mobilenet_v2']
        results = {}

        for model_name in models_to_train:
            print(f"\n{'=' * 30} {model_name} {'=' * 30}")

            # 构建模型
            model = self.build_model(model_name)

            # 训练模型
            history, best_acc = self.train(model, model_name)

            # 可视化结果
            self.visualize_training(history, model_name)

            # 记录结果
            total_params = sum(p.numel() for p in model.parameters())
            total_time = sum(history['epoch_time'])
            results[model_name] = {
                'accuracy': best_acc,
                'params': total_params,
                'time': total_time
            }

        # 4. 模型对比
        self.compare_models(results)

        print(f"\n实验完成! 所有结果文件已保存到当前目录")



def main():
    """主函数"""
    try:
        # 创建项目实例并运行实验
        project = DeepLearningProject()
        project.run_complete_experiment()

    except Exception as e:
        print(f"实验执行出错: {e}")
        print("请检查环境配置和依赖安装")


if __name__ == "__main__":
    main()