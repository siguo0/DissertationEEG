import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa.display
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import gc

data = pd.read_csv('../../../dataset/DataSet.csv')
data = data[data.target!=0]
minS_dB = []
print(data.target.value_counts())
'''
10    2479
30    2465
40    2455
20    2438
'''
bounds = np.linspace(-1, 1, 256)
norm = BoundaryNorm(bounds, ncolors=256, clip=False)

for i in tqdm(range(len(data))):
    signal = np.array(data.iloc[i][:640])
    targetDir = data.iloc[i]['target']
    # 将信号转换为浮点数，并规范化到[-1.0, 1.0]
    signal = signal.astype(np.float32) / np.max(np.abs(signal))
    # 采样率
    sr = 160  # Hz

    # 计算STFT，默认窗口大小为2048，这里我们可以选择更小的窗口，因为信号短
    n_fft = 1024  # FFT的窗口大小，可以调整以优化频率分辨率
    hop_length = len(signal)//256  # 通常取10ms
    # 执行STFT
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    print(signal)
    print(D.shape)
    # 将STFT功率谱转换为梅尔频谱图
    S = librosa.feature.melspectrogram(y=signal, sr=160, n_mels=128, win_length=128, hop_length=hop_length)

    # 将梅尔功率谱转换为对数刻度 (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # minS_dB.append(np.min(S_dB))
    S_db_normalized = (S_dB + 40) / 40

    # 绘制梅尔频谱图
    plt.figure(figsize=(20, 8))
    librosa.display.specshow(S_db_normalized, sr=sr, x_axis='time', y_axis='mel', norm=norm, cmap='viridis',fmin=0, fmax=40)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    # 确保所有图的横轴和纵轴坐标范围一致
    plt.xlim([0, 1000])
    plt.ylim([0, 40])  # 最高频率为Nyquist频率
    break
    # plt.savefig(f'/Volumes/T7 Shield/EEGdataset/DatasetImageEEG/{targetDir}/{i}.png', dpi=300)
    plt.clf()
    if i % 100 == 0:
        gc.collect()

