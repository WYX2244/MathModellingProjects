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
if __name__ == '__main__':
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