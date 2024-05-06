import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import pywt
import gc
from tqdm import tqdm
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore")

data = pd.read_csv('../../../dataset/DataSet.csv')
data = data[data.target!=0]
label = data.target
data = data.drop(['target','Task'], axis=1)
fs = 160  # 采样频率

for item in tqdm(range(len(data))):
    signalData = data.iloc[item]
    targetDir = label.iloc[item]

    data_min = signalData.min()
    data_max = signalData.max()
    signalData_normalized = (signalData - data_min) / (data_max - data_min)

    nperseg = 16   # 窗口大小
    noverlap = 8   # 窗口重叠
    # 执行STFT
    f, t, Zxx = signal.stft(signalData_normalized, fs, nperseg=nperseg, noverlap=noverlap)


    plt.figure(figsize=(20, 8))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=1)
    # plt.colorbar(label='Magnitude')
    plt.ylim([0, 40])  # 控制显示的频率范围
    plt.tight_layout()
    plt.savefig(f'/Volumes/T7 Shield/EEGdataset/TimeFrequencyEEG/{targetDir}/{item}.png', dpi=300)
    plt.clf()
    if item % 100 == 0:
        gc.collect()
