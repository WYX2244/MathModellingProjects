import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp2d
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Stickiness_ida_table:#kg.s/m4;C;-6e m2/s
    def __init__(self):
        self.df=pd.DataFrame(index=[0,30,60],columns=[0,10,20,30],data=[[179,131,100,80],[203,150,114,91],[229,168,128,102]])/100
        self.interpolator = interp2d( self.df.columns, self.df.index,self.df.values, )

    def get_ida(self,salinity,temperature):
        viscosity = self.interpolator( temperature,salinity,)
        return viscosity[0]
ida_table=Stickiness_ida_table()


def func1(v_self:np.array,loc_self:np.array,v_current:np.array,#[经度、纬度、高度]*n,
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

    if show:
        print('-'*30)
        print('v_self',v_self[0])
        print('v_current',v_current[0])
        print('v_rel',v_rel[0])
        print('ida',ida)
        print('theta',theta[0])
        print('S',S[0])
        print('a',a[0])
        print('loc_new',loc_new[0])
        print('v_new',v_new[0])
        print('-'*30)
    return v_new,loc_new



class MTKL:
    def __init__(self,num,loc_self,v_self,v_current):
        self.v_self=np.asarray([v_self]*num,dtype=float)
        self.v_current=self.v_current_init=np.asarray([v_current]*num,dtype=float)
        self.loc_self=np.asarray([loc_self]*num,dtype=float)
        self.weights=np.asarray([[1]]*num,dtype=float)
        self.num=num
    def iter(self,time_step,show=False):
        self.v_self,self.loc_self=func1(v_self=self.v_self,v_current=self.v_current,loc_self=self.loc_self,
                                   m=90000,r1=1.1,r2=19,temperature=20,salinity=1,density=1.2,time_step=time_step,show=show)
        self.v_current=np.random.multivariate_normal(mean=(0,0,0), cov=[[1, 0, 0], [0, 0.2, 0], [0, 0, 0]]  , size=self.num)
    def message(self):
        return

class Search_Action():
    def __init__(self,num_searcher,max_range=3000,speed=5,scan_time=60*30,worktime=8*60*60):

        self.num_searcher=num_searcher
        self.max_range=max_range
        self.speed=speed
        self.move_time=max_range*2/speed
        self.scan_time=scan_time
        self.mtkl = MTKL(100, [0, 0, 0], [0, 0, 0], [0.5, 0.05, 0])
        self.real_loc=MTKL(1, [0, 0, 0], [0, 0, 0], [0.5, 0.05, 0])
        self.n_place_for_one_searcher=int(worktime//(self.move_time+scan_time))
        self.searcher_plan=pd.DataFrame([],index=range(self.num_searcher),columns=range(self.n_place_for_one_searcher))

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
            stacked_array = np.vstack([j for i in sc.searcher_plan.values for j in i if type(j) is not float])
            reward=self.reward_bayes(stacked_array)
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


    def iter_planning_global(self,start_column=1,max_iter=10,max_div=1,depth=2):#该搜索器执行规划并重置其目的地列表
        i=0
        while i<max_iter:
            df=self.searcher_plan.copy()
            for idx in df.index:
                for st in range(start_column,self.n_place_for_one_searcher,depth-1):
                    _,sr_idx=self.iter_planning_personnal(idx,start_column=st)
                    self.searcher_plan.loc[idx]=sr_idx

            if i>0 :
                div = abs(df.values-self.searcher_plan.values)
                div=np.sum(div)
                div = np.linalg.norm(div)
                print(i, div)
                plt.scatter(self.mtkl.loc_self[:, 0], self.mtkl.loc_self[:, 1], label='All Points')
                # 遍历 sr，并在图上标记这些点
                for index, value in self.searcher_plan.iloc[0].dropna().items():
                    plt.scatter(value[0], value[1], color='red')  # 使用红色标记这些点
                    plt.text(value[0], value[1], str(index))  # 可以在点旁边标记索引
                for index, value in self.searcher_plan.iloc[1].dropna().items():
                    plt.scatter(value[0], value[1], color='green')  # 使用红色标记这些点
                    plt.text(value[0], value[1], str(index))  # 可以在点旁边标记索引
                plt.show()
                if div<max_div:
                    return
            i+=1
        return


    def iter_day(self):
        for i in range(24*60):
            self.mtkl.iter(60)
            self.real_loc.iter(60)
        self.init_loc()
        self.iter_planning_global()

        stacked_array = np.vstack([j for i in sc.searcher_plan.values for j in i if type(j) is not float])
        prob_today=self.prob_found(stacked_array,self.mtkl.loc_self)
        self.mtkl.weights*=(1-prob_today)
        prob_real_found=self.prob_found(stacked_array,self.real_loc.loc_self)
        print(self.real_loc.loc_self.shape,prob_real_found.shape)
        print(prob_real_found)
        return prob_today

    def init_center(self, n_iterations=10):
        # 初始化K-means，只有一个聚类中心
        kmeans = KMeans(n_clusters=self.num_searcher, max_iter=n_iterations)
        # 聚类算法每次迭代都会尝试找到最佳位置
        kmeans.fit(self.mtkl.loc_self, sample_weight=self.mtkl.weights[:,0])
        locs = kmeans.cluster_centers_
        return locs

    def reward_bayes(self,planning:np.ndarray):#一个堆叠后的计划探索地点列表以获取总奖励
        return np.sum(self.prob_found(planning,self.mtkl.loc_self)*self.mtkl.weights)
    def prob_found(self,planning,locs):
        distance_matrix = np.linalg.norm(locs[:, np.newaxis, :] - planning[np.newaxis, :, :], axis=2)
        # 计算概率
        prob_matrix = 1 - (distance_matrix / self.max_range)
        prob_matrix = np.clip(prob_matrix, 0, 1)  # 确保概率值在 0 和 1 之间
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



if __name__ == '__main__':
    sc=Search_Action(2)
    sc.iter_day()
    print('go')
    sc.init_loc()
    a=sc.iter_planning_global()

