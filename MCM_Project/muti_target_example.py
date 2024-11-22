import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp2d
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from concurrent.futures import  ThreadPoolExecutor,as_completed,ProcessPoolExecutor
warnings.filterwarnings("ignore", category=DeprecationWarning)
import netCDF4 as nc


class Stickiness_ida_table:#kg.s/m4;C;-6e m2/s
    def __init__(self):
        self.df=pd.DataFrame(index=[0,30,60],columns=[0,10,20,30],data=[[179,131,100,80],[203,150,114,91],[229,168,128,102]])/100
        self.interpolator = interp2d( self.df.columns, self.df.index,self.df.values, )

    def get_ida(self,salinity,temperature):
        viscosity = self.interpolator( temperature,salinity,)
        return viscosity[0]

class Bathymetric_table:
    def __init__(self):
        data = nc.Dataset('gebco_2023_n28.0664_s20.6836_w-97.3022_e-82.8296.nc')
        self.lats = np.asarray(data.variables['lat'][:])
        self.lons = np.asarray(data.variables['lon'][:])
        self.elevation = np.asarray(data.variables['elevation'][:])

    def get_depths(self, coords):
        lat_indices = np.abs(self.lats[:, None] - coords[:, 0]).argmin(axis=0)
        lon_indices = np.abs(self.lons[:, None] - coords[:, 1]).argmin(axis=0)
        return self.elevation[lat_indices, lon_indices]

ida_table=Stickiness_ida_table()
bathy_table=Bathymetric_table()

def func1(v_self:np.array,loc_self:np.array,v_current:np.array,#[纬度、经度、高度]*n,
          m,r1,r2,
          temperature,salinity,density,
          time_step:float,
          show=False):

    v_rel=v_current-v_self
    ida=ida_table.get_ida(salinity,temperature)#查ida
    theta=np.arctan((v_current[:,[1]]-v_self[:,[1]])/(v_current[:,[0]]-v_self[:,[0]]))#求theta角
    S=np.pi*r1*np.sqrt(np.sin(theta)**2*r2**2+np.cos(theta)**2*r1**2)
    a=(S*v_rel)/(2*m)*(density*np.linalg.norm(v_rel,axis=1).reshape(-1,1)+4*ida/(r1+r2))
    loc_new=loc_self+v_self*time_step+0.5*a*time_step**2
    v_new=v_self+a*time_step

    return v_new,loc_new



class Muti_MTKL:
    def __init__(self,num,num_submarine,loc_self_ll,v_self,v_current,beta_extra=1.0):#n,纬度，经度，高度
        self.v_self=np.asarray([v_self]*num,dtype=float)
        self.v_current=self.v_current_init=np.asarray([v_current]*num,dtype=float)
        self.loc_self=np.asarray([[[0,0,0]]*num_submarine]*num,dtype=float)
        self.loc_self_ll=np.asarray([loc_self_ll]*num,dtype=float)
        self.loc_self_ll_init = np.asarray([loc_self_ll]*num,dtype=float)
        self.weights=np.asarray([[[1]]*num_submarine]*num,dtype=float)
        self.num=num
        self.is_dead=np.asarray([[[False]]*num_submarine]*num,dtype=bool)
        self.beta_extra=beta_extra
        self.num_submarine=num_submarine
    def iter(self,time_step,step,show=False,weight_known=0.5,):
        last_loc_self=self.loc_self.copy()
        for i in range(self.num_submarine):
            self.v_self[:,i],self.loc_self[:,i]=func1(v_self=self.v_self[:,i],v_current=self.v_current[:,i],loc_self=self.loc_self[:,i],
                                       m=90000,r1=1.1,r2=19,temperature=13.909364,salinity=38.7927,density=1.06,time_step=time_step,show=show)
            self.v_self[:,i][self.is_dead[:,i, 0]] = 0
            # 将死亡个体的位置替换为最后的位置
            self.loc_self[:,i][self.is_dead[:,i, 0]] = last_loc_self[:,i][self.is_dead[:,i, 0]]
            s=(self.loc_self_ll_init[:,i]+self.loc_self[:,i]/(111100*np.asarray([1,np.cos(self.loc_self_ll_init[0,i,0]/180*np.pi),1,])))
            self.loc_self_ll[:, i]=s

            depths=bathy_table.get_depths(self.loc_self_ll[:,i,:2])
            self.is_dead[:,i]=(depths>self.loc_self_ll[:,i,2]).reshape(-1,1)
            if step==0:
                self.v_current[:,i]=self.v_current_init[:,i]*weight_known\
                           +(1-weight_known)*np.random.multivariate_normal(mean=(0,0,0), cov=np.asarray([[0.00831335 ,0.00186226,0],[0.00186226,0.0109959,0],[0,0,0]])*self.beta_extra  , size=self.num)


class Search_Action():
    def __init__(self,num_searcher,max_range=3000,min_range=100,speed=5,scan_time=60*30,worktime=8*60*60
                 ,num_mtkl=100,loc_self_ll=[0,0,0],v_self=[0,0,0],v_current=[0.000001,0.0000001,0],beta_extra=1,scaling=0):
        self.num_searcher=num_searcher
        self.max_range=max_range
        self.speed=speed
        self.move_time=max_range*2/speed
        self.scan_time=scan_time
        self.mtkl = Muti_MTKL(num_mtkl,loc_self_ll,v_self,v_current,beta_extra=beta_extra)
        self.real_loc=Muti_MTKL(1, loc_self_ll, v_self, v_current,)
        self.n_place_for_one_searcher=int(worktime//(self.move_time+scan_time))
        self.searcher_plan=pd.DataFrame([],index=range(self.num_searcher),columns=range(self.n_place_for_one_searcher))
        self.cumulative_prob=0
        self.cumulative_unfound_prob=1
        self.min_range=min_range
        self.scaling=scaling
        self.day=0
        self.km=None
    def init_loc(self,):#根据当天的蒙特卡洛状态初始化搜索器位置
        self.searcher_plan=pd.DataFrame([],index=range(self.num_searcher),columns=range(self.n_place_for_one_searcher))
        loc=self.init_center()
        for i in self.searcher_plan.index:
            self.searcher_plan.iloc[i,0]=loc[i]

        return

    def iter_planning_personnal(self,idx,sr_idx=None,start_column=1,depth_to_go=2):
        if sr_idx is None:
            sr_idx=self.searcher_plan.iloc[idx]
        sr_rewards=pd.Series(index=[i*np.pi/6 for i in range(12)])
        depth_to_go=min(depth_to_go,self.n_place_for_one_searcher-start_column)

        if not depth_to_go:
            self.searcher_plan.loc[idx]=sr_idx
            stacked_array = np.vstack([j for i in self.searcher_plan.values for j in i if type(j) is not float])
            reward=self.reward_bayes(stacked_array,scaling=self.scaling)
            return reward,sr_idx

        sr_idx_best=None
        best_reward=-np.inf
        for angle in sr_rewards.index:
            sr_idx[start_column]=sr_idx[start_column-1]+np.asarray([self.max_range*2*np.sin(angle),self.max_range*1.8*np.cos(angle),0])
            reward,sr_idx=self.iter_planning_personnal(idx,sr_idx,start_column+1,depth_to_go-1)
            if reward>best_reward:
                best_reward=reward
                sr_idx_best=sr_idx.copy()
        return best_reward,sr_idx_best


    def iter_planning_global(self,start_column=1,max_iter=3,max_div=1,depth=2):#该搜索器执行规划并重置其目的地列表
        i=0
        best=0
        best_plan=None
        last_reward=0
        while i<max_iter:
            df=self.searcher_plan.copy()
            for idx in df.index:
                for st in range(start_column,self.n_place_for_one_searcher,depth-1):
                    reward,sr_idx=self.iter_planning_personnal(idx,start_column=st)
                    self.searcher_plan.loc[idx]=sr_idx
                    if reward>best:
                        best_plan=self.searcher_plan.copy()
                        best=reward
                # plt.scatter(self.mtkl.loc_self[:, 0], self.mtkl.loc_self[:, 1], label='All Points')
                # # 遍历 sr，并在图上标记这些点
                # for index, value in self.searcher_plan.iloc[0].dropna().items():
                #     plt.scatter(value[0], value[1], color='red')  # 使用红色标记这些点
                #     plt.text(value[0], value[1], str(index))  # 可以在点旁边标记索引
                # for index, value in self.searcher_plan.iloc[1].dropna().items():
                #     plt.scatter(value[0], value[1], color='green')  # 使用红色标记这些点
                #     plt.text(value[0], value[1], str(index))  # 可以在点旁边标记索引
                # plt.show()
            i+=1
        self.searcher_plan=best_plan
        return best,best_plan


    def iter_day(self):
        for i in range(24*60):
            self.mtkl.iter(60,step=i,weight_known=0.5 if self.day==0 else 0.3 if self.day==1 else 0.1 if self.day==2 else 0)
            self.real_loc.iter(60,step=i,weight_known=0.5 if self.day==0 else 0.3 if self.day==1 else 0.1 if self.day==2 else 0)
        self.init_loc()
        self.iter_planning_global()

        self.stacked_array = np.vstack([j for i in self.searcher_plan.values for j in i])
        prob_today=self.prob_found(self.stacked_array,self.mtkl.loc_self)
        self.mtkl.weights*=(1-prob_today)
        prob_real_found=self.prob_found(self.stacked_array,self.real_loc.loc_self)
        self.cumulative_unfound_prob*=(1-prob_real_found)
        self.cumulative_prob=1-self.cumulative_unfound_prob
        self.day+=1
        return prob_real_found,self.cumulative_prob

    def init_center(self, n_iterations=10):
        # 初始化K-means，只有一个聚类中心
        kmeans = KMeans(n_clusters=self.num_searcher, max_iter=n_iterations)
        # 聚类算法每次迭代都会尝试找到最佳位置
        kmeans.fit(self.mtkl.loc_self, sample_weight=self.mtkl.weights[:,0])
        locs = kmeans.cluster_centers_
        self.km=kmeans
        return locs

    def reward_bayes(self,planning:np.ndarray,scaling=0):#一个堆叠后的计划探索地点列表以获取总奖励
        return np.sum(self.prob_found(planning,self.mtkl.loc_self,scaling=scaling)*self.mtkl.weights,)
    def prob_found(self,planning,locs,scaling=0):
        distance_matrix = np.linalg.norm(locs[:, np.newaxis, :] - planning[np.newaxis, :, :], axis=2)
        # 计算概率
        prob_matrix=np.exp(-(distance_matrix-self.min_range)/750)**(1-scaling)

        if  scaling==0:
            prob_matrix[distance_matrix>self.max_range]=0

        prob_matrix = np.clip(prob_matrix, 0, 0.99-scaling)  # 确保概率值在 0 和 1 之间
        # 使用 weights 调整每个 loc 点的概率
        rewards_matrix = prob_matrix
        not_found_prob = np.prod(1 - rewards_matrix, axis=1)
        # 计算至少有一个搜索器找到每个目标的总概率
        rewards_matrix = 1 - not_found_prob
        # 将结果转换为 3x1 的数组
        rewards_matrix = rewards_matrix.reshape(-1, 1)
        return rewards_matrix
    def reward_a_evaluation(self):
        return


def go_action(num_searcher,beta_extra=1,scaling=0,show=False):
    sc=Search_Action(num_searcher,max_range=3000,min_range=300,speed=5,scan_time=60*30,worktime=8*60*60
                 ,num_mtkl=100,loc_self_ll=[38,20,-1000],v_self=[0,0,0],v_current=[0.00001,0.000001,0.0],beta_extra=beta_extra,scaling=scaling)
    for i in range(14):
        prob=sc.iter_day()[1][0][0]
        print(f'{num_searcher} {beta_extra} {scaling} day{i}:',prob)
        if prob>0.95:
            return [prob,i + 1,num_searcher,beta_extra,scaling,True]  # day,prob

        if show:
            plt.scatter(sc.mtkl.loc_self[:, 0], sc.mtkl.loc_self[:, 1], color='blue')
            plt.scatter(sc.real_loc.loc_self[:, 0], sc.real_loc.loc_self[:, 1], color='red')
            plt.scatter(sc.stacked_array[:, 0], sc.stacked_array[:, 1], color='green')
            plt.show()

    return [prob,i,num_searcher,beta_extra,scaling,False]

def dual_mtkl():
    tp=ProcessPoolExecutor(8)
    fs=[]
    values=[]
    for i in range(1000):
        fs.append(tp.submit(go_action,2))
        fs.append(tp.submit(go_action,3))
        fs.append(tp.submit(go_action,4))
        fs.append(tp.submit(go_action,5))
        fs.append(tp.submit(go_action,6))
        fs.append(tp.submit(go_action,7))
        fs.append(tp.submit(go_action,8))
        fs.append(tp.submit(go_action,5,1.2))
        fs.append(tp.submit(go_action,5,1.5))
        fs.append(tp.submit(go_action,5,1,0.1))
        fs.append(tp.submit(go_action,5,1,0.2))
    for result in as_completed(fs):
        result=result.result()
        print(result)
        values.append(result)
        df=pd.DataFrame(values,columns=['prob','day','num_searcher','beta_extra','scaling','>0.95'])
        df.to_excel('1.xlsx')

def mtkl_3d_result(mtkl):
    # 确定坐标点覆盖的范围
    min_lon, max_lon = np.min(mtkl.loc_self_ll[:,:, 1]), np.max(mtkl.loc_self_ll[:,:, 1])
    min_lat, max_lat = np.min(mtkl.loc_self_ll[:,:, 0]), np.max(mtkl.loc_self_ll[:,:, 0])

    # 扩展这个范围以确保显示效果（可根据需要调整）
    delta_lon = (max_lon - min_lon) * 2
    delta_lat = (max_lat - min_lat) * 2

    # 筛选地形数据
    lon_mask = (bathy_table.lons >= min_lon - delta_lon) & (bathy_table.lons <= max_lon + delta_lon)
    lat_mask = (bathy_table.lats >= min_lat - delta_lat) & (bathy_table.lats <= max_lat + delta_lat)

    # 创建筛选后的网格
    X, Y = np.meshgrid(bathy_table.lons[lon_mask], bathy_table.lats[lat_mask])
    Z = bathy_table.elevation[np.ix_(lat_mask, lon_mask)]
    print(X, Y, Z)
    # 绘制3D地形图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    for kkk in range(mtkl.num_submarine):
        color=colors[kkk]
        # 添加坐标点

        ax.scatter(mtkl.loc_self_ll[:,kkk, 1], mtkl.loc_self_ll[:,kkk, 0], mtkl.loc_self_ll[:,kkk, 2] + 100, color=color)
        ax.set_ylabel('Longitude')
        ax.set_xlabel('Latitude')
        ax.set_zlabel('Depth')
        ax.set_title(f'Day {h}')
    plt.show()

def show_init_place_by_kmeans(mtkl:Muti_MTKL):
    kmeans = KMeans(n_clusters=5, max_iter=10)
    # 聚类算法每次迭代都会尝试找到最佳位置
    kmeans.fit(mtkl.loc_self_ll.reshape(-1,3), sample_weight=mtkl.weights.reshape(-1))
    X = mtkl.loc_self_ll[:, :,:2].reshape(-1,2)
    centers = kmeans.cluster_centers_[:, :2]  # 同样只取前两维
    # 绘制样本点
    cluster_labels = kmeans.labels_
    # 您的颜色列表
    # 确保颜色列表的长度与簇的数量相匹配
    num_clusters = len(set(cluster_labels))
    assert len(colors) >= num_clusters, "颜色列表长度必须至少与簇的数量相同"
    # 为每个数据点分配颜色
    point_colors = [colors[label] for label in cluster_labels]
    plt.scatter(X[:, 0], X[:, 1], c=point_colors, cmap='viridis', marker='o')
    # 绘制聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='x')
    plt.show()
    return kmeans.cluster_centers_








def show_planning(mtkl):
    global searcher_plan
    loc=show_init_place_by_kmeans(mtkl)

    for i in searcher_plan.index:
        searcher_plan.iloc[i, 0] = loc[i]
    def reward_bayes(planning:np.ndarray,scaling=0):#一个堆叠后的计划探索地点列表以获取总奖励
        return np.sum(prob_found(planning,mtkl.loc_self_ll.reshape(-1,3),scaling=scaling)*mtkl.weights.reshape(-1,1),)
    def prob_found(planning,locs,scaling=0):
        distance_matrix = np.linalg.norm(locs.reshape(-1,3)[:, np.newaxis, :] - planning[np.newaxis, :, :], axis=2)
        # 计算概率
        prob_matrix=np.exp(-(distance_matrix-300/111100)/750)**(1-scaling)

        if  scaling==0:
            prob_matrix[distance_matrix>3000/111100]=0

        prob_matrix = np.clip(prob_matrix, 0, 0.99-scaling)  # 确保概率值在 0 和 1 之间
        # 使用 weights 调整每个 loc 点的概率
        rewards_matrix = prob_matrix
        not_found_prob = np.prod(1 - rewards_matrix, axis=1)
        # 计算至少有一个搜索器找到每个目标的总概率
        rewards_matrix = 1 - not_found_prob
        # 将结果转换为 3x1 的数组
        rewards_matrix = rewards_matrix.reshape(-1, 1)
        return rewards_matrix
    def iter_planning_personnal(idx,sr_idx=None,start_column=1,depth_to_go=2):
        if sr_idx is None:
            sr_idx=searcher_plan.iloc[idx]
        sr_rewards=pd.Series(index=[i*np.pi/6 for i in range(12)])
        depth_to_go=min(depth_to_go,9-start_column)

        if not depth_to_go:
            searcher_plan.loc[idx]=sr_idx
            stacked_array = np.vstack([j for i in searcher_plan.values for j in i if type(j) is not float])
            reward=reward_bayes(stacked_array,scaling=0)
            return reward,sr_idx

        sr_idx_best=None
        best_reward=-np.inf
        for angle in sr_rewards.index:
            sr_idx[start_column]=sr_idx[start_column-1]+np.asarray([3000/111100*2*np.sin(angle),3000/111100*1.8*np.cos(angle),0])
            reward,sr_idx=iter_planning_personnal(idx,sr_idx,start_column+1,depth_to_go-1)
            if reward>best_reward:
                best_reward=reward
                sr_idx_best=sr_idx.copy()
        return best_reward,sr_idx_best


    def iter_planning_global(start_column=1,max_iter=3,max_div=1,depth=2):#该搜索器执行规划并重置其目的地列表
        global searcher_plan
        i=0
        best=0
        best_plan=None
        last_reward=0
        while i<max_iter:
            df=searcher_plan.copy()
            for idx in df.index:
                for st in range(start_column,9,depth-1):
                    reward,sr_idx=iter_planning_personnal(idx,start_column=st)
                    searcher_plan.loc[idx]=sr_idx
                    if reward>best:
                        best_plan=searcher_plan.copy()
                        best=reward
            i+=1
        searcher_plan=best_plan
        return best,best_plan

    iter_planning_global()
    print(mtkl.loc_self_ll[:,0,0])
    plt.scatter(mtkl.loc_self_ll[:,0, 0], mtkl.loc_self_ll[:,0, 1], label='Subm 0',c='grey')
    plt.scatter(mtkl.loc_self_ll[:,1, 0], mtkl.loc_self_ll[:,1, 1], label='Subm 1',c='black')

    # 遍历 sr，并在图上标记这些点
    for sridx in range(5):
        for index, value in searcher_plan.iloc[sridx].dropna().items():
            plt.scatter(value[0], value[1], color=colors[sridx])  # 使用红色标记这些点
            plt.text(value[0], value[1], str(index))  # 可以在点旁边标记索引

    plt.show()



if __name__ == '__main__':
    searcher_plan=pd.DataFrame([],index=range(5),columns=range(9))

    colors = ['blue','red',  'yellow','pink', 'green', ]
    mtkl=Muti_MTKL(100,1,[[26,-93,-1000],[26.1,-93.1,-800]],[[0,0,0],[0,0,0]],[[0.00001,0.00001,0],[0.00001,0.00001,0]])
    for h in range(3):
        for i in range(24*60):
            mtkl.iter(60,i)
        mtkl_3d_result(mtkl)

