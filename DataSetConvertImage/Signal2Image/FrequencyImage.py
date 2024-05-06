import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings("ignore")

data = pd.read_csv('../../../dataset/DataSet.csv')
data = data[data.target!=0]
label = data.target
data = data.drop(['target','Task'], axis=1)

for i in tqdm(range(len(data))):
    targetDir = label.iloc[i]
    item = data.iloc[i]
    voltage = item.values
    time = list(range(len(item.index)))
    # 计算傅里叶变换
    fft_values = np.fft.fft(voltage)

    # 计算频率轴
    n = len(time)
    frequency = np.fft.fftfreq(n, d=time[1] - time[0])

    # 取正频率部分
    positive_frequencies = frequency[:n//2]
    magnitudes = np.abs(fft_values)[:n//2]  # 取模以获取幅值

    # 绘制频率图
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, magnitudes)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency in Hertz [Hz]')
    plt.ylabel('Amplitude')
    plt.ylim([0,0.01])
    plt.grid(True)
    plt.savefig(f'/Volumes/T7 Shield/EEGdataset/FrequencyImage/{targetDir}/{i}.png', dpi=300)
    plt.clf()
    if i % 100 == 0:
        gc.collect()
