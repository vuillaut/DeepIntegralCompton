import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

def load_data(filename):
    array = np.load(filename)
    header = ['e1', 'e2', 'yy1', 'zz1', 'yy2', 'zz2', 'y1', 'z1', 'y2', 'z2']
    df = pd.DataFrame(array, columns=header)
    tp  = Path(filename).with_suffix('').name.split('_')
    theta = int(tp[1])
    phi = int(tp[3])
    return df, theta, phi



def hist2d_plans(data, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    
    axes[0].set_title('ISGRI')
    _, _, _, cm = axes[0].hist2d(data.y1, data.z1, **kwargs)
    fig.colorbar(cm, ax=axes[0])
    axes[1].set_title('PICSIT')
    _, _, _, cm = axes[1].hist2d(data.y2, data.z2, **kwargs)
    fig.colorbar(cm, ax=axes[1])
    
    for ax in axes.ravel():
        ax.set_xlabel('y / cm')
        ax.set_ylabel('z / cm')
    
    return axes

def scatter_plans(data, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    
    axes[0].set_title('ISGRI')
    cm = axes[0].scatter(data.y1, data.z1, c=data.e1, **kwargs)
    fig.colorbar(cm, ax=axes[0])
    
    axes[1].set_title('PICSIT')
    cm = axes[1].scatter(data.y2, data.z2, c=data.e2, **kwargs)
    fig.colorbar(cm, ax=axes[1])
    
    for ax in axes.ravel():
        ax.set_xlabel('y / cm')
        ax.set_ylabel('z / cm')
    
    return axes

def scatter_plans_2(data, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    
    axes[0].set_title('ISGRI')
    cm = axes[0].scatter(data.yy1, data.zz1, c=data.e1, **kwargs)
    fig.colorbar(cm, ax=axes[0])
    
    axes[1].set_title('PICSIT')
    cm = axes[1].scatter(data.yy2, data.zz2, c=data.e2, **kwargs)
    fig.colorbar(cm, ax=axes[1])
    
    for ax in axes.ravel():
        ax.set_xlabel('y / cm')
        ax.set_ylabel('z / cm')
    
    return axes

