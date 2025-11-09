import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os

# 优化中文字体设置：尝试按顺序使用可用字体，避免 SimHei 不存在时大量警告。
# 回退顺序：SimHei -> Noto Sans CJK SC -> WenQuanYi Micro Hei -> DejaVu Sans
_preferred_fonts = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"]
_system_fonts = {os.path.basename(f.fname).split(".")[0]: f.fname for f in fm.fontManager.ttflist}

def _choose_chinese_font(preferred_list):
    for name in preferred_list:
        # 直接匹配字体家族名称（family）
        for font in fm.fontManager.ttflist:
            if font.name == name:
                return font.name
        # 再尝试文件名前缀粗匹配
        for sys_name, path in _system_fonts.items():
            if name.lower().replace(" ", "") in sys_name.lower().replace(" ", ""):
                return name  # 让 matplotlib 再次解析以获取 family
    return None

_chosen_font = _choose_chinese_font(_preferred_fonts)
if _chosen_font:
    plt.rcParams['font.sans-serif'] = [_chosen_font]
else:
    # 没有找到中文字体，使用默认并给出一次性提示。
    warnings.warn("未检测到常用中文字体，将使用默认英文字体。可安装 'fonts-noto-cjk' 或 'wenquanyi' 包以改善显示。", RuntimeWarning)

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块

# 抑制重复的 findfont 警告信息
warnings.filterwarnings("once", category=UserWarning, module="matplotlib.font_manager")


class TrainingVisualizer:
    def __init__(self, save_dir="training_plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.rewards = []
        self.losses = []
        self.learning_rates = []
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 15))
        self.fig.tight_layout(pad=5.0)

    def add_epoch_data(self, avg_reward, avg_loss, current_lr):
        """在每个轮次结束后添加数据。"""
        self.rewards.append(avg_reward)
        self.losses.append(avg_loss)
        self.learning_rates.append(current_lr)

    def plot(self, epoch):
        """绘制并保存图表。"""
        epochs = range(1, len(self.rewards) + 1)

        # 绘制平均奖励
        self.axs[0].clear()
        self.axs[0].plot(epochs, self.rewards, 'b-o', label='平均奖励')
        self.axs[0].set_title('每轮的平均奖励')
        self.axs[0].set_xlabel('轮次')
        self.axs[0].set_ylabel('奖励')
        self.axs[0].grid(True)
        self.axs[0].legend()

        # 绘制平均损失
        self.axs[1].clear()
        self.axs[1].plot(epochs, self.losses, 'r-o', label='平均损失')
        self.axs[1].set_title('每轮的平均损失')
        self.axs[1].set_xlabel('轮次')
        self.axs[1].set_ylabel('损失')
        self.axs[1].grid(True)
        self.axs[1].legend()

        # 绘制学习率
        self.axs[2].clear()
        self.axs[2].plot(epochs, self.learning_rates, 'g-o', label='学习率')
        self.axs[2].set_title('每轮的学习率')
        self.axs[2].set_xlabel('轮次')
        self.axs[2].set_ylabel('学习率')
        self.axs[2].grid(True)
        self.axs[2].legend()

        # 保存图表
        save_path = os.path.join(self.save_dir, f"training_progress_epoch_{epoch}.png")
        plt.savefig(save_path)
        
    def close(self):
        """关闭matplotlib绘图窗口。"""
        plt.close(self.fig)
