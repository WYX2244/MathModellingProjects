import time

import pandas as pd
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import requests
import os
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor
def download_file(url, folder, filename=None):
    # 如果未指定文件名，则从 URL 中提取
    if filename is None:
        filename = url.split('/')[-1]

    # 创建完整的文件路径
    file_path = os.path.join(folder, filename)

    # 发送请求获取文件内容
    try:
        print(url)
        response = requests.get(url,verify=False)
        response.raise_for_status()  # 确保请求成功
    except requests.RequestException as e:
        print(f"请求错误: {e}")
        return
    # 将文件内容写入指定路径
    try:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已保存到: {file_path}")
    except IOError as e:
        print(f"文件写入错误: {e}")

def download():
    tp = ThreadPoolExecutor(max_workers=1024)
    for url in df['Filename of the Station'].tolist():
        tp.submit(download_file, *(url, 'moored_current_nc'))
    exit()







def salty_tempy():
    df=pd.read_csv('ocldb1706832809.15338.csv',header=None,)
    df.columns=['depth','temp','salt']
    df=df[~df['salt'].isna()]
    df=df[df['depth']>600]

def speedy():
    df = pd.read_csv('Moored Current Meters.csv')
    for c in ['Latitude Range(+deg_N)', 'Longitude Range(+deg_E)', 'Instrument Depth(m)']:
        df[c] = df[c].apply(lambda x: float(x.split('~')[0]))
    df = df[df['Instrument Depth(m)'] > 600]
    # df['loc'] = df.apply(lambda x: str(x['Latitude Range(+deg_N)']) + ',' + str(x['Longitude Range(+deg_E)']), axis=1)
    df = df[(df['Latitude Range(+deg_N)'] < 60) & (df['Latitude Range(+deg_N)'] > 30)]
    # df=df[df['Longitude Range(+deg_E)']==4.998]
    us = []
    vs = []
    for f in df['Filename of the Station'].tolist():
        f = 'moored_current_nc/' + f.split('/')[-1]
        data = nc.Dataset(f)
        u = list(data.variables['u'][:])
        v = list(data.variables['v'][:])
        us += u
        vs += v

    speeds = np.asarray([vs, us]).transpose()
    print(speeds.shape)
    speeds = speeds[~np.isnan(speeds).any(axis=1)]
    print(speeds.shape)
    speeds = speeds.transpose()
    print(speeds.shape)
    print(np.cov(speeds))
    print(np.mean(speeds, axis=1))

def bathy():
    dataset=nc.Dataset('gebco_2023_n39.873_s31.2598_w13.8208_e25.7446.nc')
    print(dataset.variables)
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]
    elevation = dataset.variables['elevation'][:]

    # 创建经纬度的网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    # 将网格和高程数据展平
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    elevation_flat = elevation.flatten()

    # 创建DataFrame
    df = pd.DataFrame({
        'Latitude': lat_flat,
        'Longitude': lon_flat,
        'Elevation': elevation_flat
    })
    print(df)

def mtkl_result():
    df = pd.read_excel('1.xlsx')
    print(df)
    df = df[df['beta_extra'] == 1]
    df = df[df['scaling'] == 0]
    num_searcher = df['num_searcher'].unique().tolist()
    plt.bar(num_searcher,
            [sum(df[df['num_searcher'] == i]['>0.95']) / df[df['num_searcher'] == i]['>0.95'] for i in num_searcher])
    plt.show()
    time.sleep(100)
def ree():
    df = pd.read_csv('ocldb1706832809.15338.csv', header=None, )
    df.columns = ['depth', 'temp', 'salt']
    df = df[~df['salt'].isna()]
    plt.scatter(df['depth'], df['temp'],alpha=0.1,s=3)
    plt.xlabel('Depth')
    plt.ylabel('Temperature')

    plt.show()
    plt.scatter(df['depth'], df['salt'],alpha=0.1,s=3)
    plt.xlabel('Depth')
    plt.ylabel('Salinity')
    plt.show()
    time.sleep(111)
def mean_of_arrays(series):
    a = np.asarray([[i[0] for i in series.values], [i[1] for i in series.values]])
    return np.mean(a, axis=1)
def re_ana1():
    df = pd.read_excel('洋流速度.xlsx')
    df['center'] = df['center'].apply(lambda x: [i for i in x[1:-1].split(' ') if '.' in i])
    df['center'] = df['center'].apply(lambda x: [float(i) for i in x])

    df = df.groupby(['day', 'v洋流', 'v潜艇'], as_index=False).agg({
        'R': 'mean',  # 对普通数值列使用内置的 mean 函数
        'center': mean_of_arrays  # 对数组列使用自定义的聚合函数
    })
    df['center_纬度方向(m)'] = df['center'].apply(lambda x: x[0])
    df['center_经度方向(m)'] = df['center'].apply(lambda x: x[1])

    print(df)
    df.to_excel('11.xlsx')
if __name__ == '__main__':




    df=pd.read_excel('1.xlsx',index_col=0)
    print(df)
    df=df.drop('prob',axis=1)
    df['>0.95']=df['>0.95'].astype(int)
    grouped = df.groupby(['num_searcher', 'beta_extra', 'scaling'])


    # 计算每个day的累计成功率
    def cumulative_success_rate(group):
        # 按day排序
        group = group.sort_values(by='day')
        # 计算累计成功的数量
        group['cumulative_success'] = group['>0.95'].cumsum()
        # 计算累计成功率
        group['cumulative_success_rate'] = group['cumulative_success'] / len(group)
        group=group.groupby('day').last()
        return group


    # 应用函数到每个分组
    cumulative_success_df = grouped.apply(cumulative_success_rate)
    print(cumulative_success_df)
    cumulative_success_df.to_excel('re1.xlsx')

    # 展示结果
