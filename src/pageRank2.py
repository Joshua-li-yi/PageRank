import timeit
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import threading
from numba import jit
# 设置参数
Beta = 0.85
derta = 0.001
all_line = 103690
# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 从txt导入数据、将数据转化成 csv 格式 nodes 输入和输出，类似于将边存起来
# 输出nodes--------dataframe 格式 和all_node --------list
def load_data(filePath, output_csv=False):
    txt = np.loadtxt(filePath)
    nodes = pd.DataFrame(data=txt,columns=['input_node', 'output_node'])
    if output_csv is True:
        nodes.to_csv('WikiData.csv')
    # 根据inputpage的值排序
    nodes.sort_values('input_node', inplace=True)
    # 重置索引
    nodes.reset_index(inplace=True, drop=True)
    # all_node 加载为list 有重复值
    all_node = nodes['input_node'].values.tolist()
    all_node.extend(nodes['output_node'].values.tolist())
    # all_node 转为set 再转为list
    all_node = set(all_node)
    all_node = list(all_node)
    # all_note 升序排列
    all_node.sort()
    # print(all_node)
    # print(nodes)
    return nodes, all_node


# 生成rank值
def generate_rank(all_node):
    # 初始rank
    rank = pd.DataFrame({'page': all_node, 'score': 1 / len(all_node)}, columns=['page', 'score'])
    # 这个有点问题，得查查怎么改
    # tqdm.pandas(desc="rank initial")
    # rank.progress_apply(lambda x: x ** 2)
    # 将page列设置为索引
    rank.set_index('page', inplace=True)
    return rank


# 将之前得nodes 存起来得边，转化为矩阵。用的是老师PPT上的'source_node','degree','destination_nodes'结构
def nodes_to_M(nodes):
    M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
    # 将M的source_node列设置为索引
    M.set_index('source_node', inplace=True)
    with tqdm(total=nodes.shape[0], desc='M matrix generate progress') as bar:
        for index, node_row in nodes.iterrows():
            input_node = node_row[0]
            output_node = node_row[1]
            if input_node not in M.index.tolist():
                M.loc[input_node, 'degree'] = int(1)
                M.loc[input_node, 'destination_nodes'] = np.array([output_node])
            else:
                M.loc[input_node, 'degree'] += 1
                M.loc[input_node, 'destination_nodes'] = np.append(M.loc[input_node, 'destination_nodes'], output_node)
            bar.update(1)
    return M


# 将一个列表划分为多个小列表

def list_to_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) * per_list_len)
    end_list = [list(i) for i in list_of_group]  # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count != 0 else end_list
    return end_list


# block_strip algorithm


def block_strip(M, block_node_groups):
    # 存最后的各个划分后的M
    M_block_stripe = []
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            temp_block_M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
            temp_block_M.set_index('source_node', inplace=True)
            # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
            for per_node in node_group:
                for index, row in M.iterrows():
                    if per_node in row['destination_nodes'].tolist():
                        if per_node not in temp_block_M.index.tolist():
                            temp_block_M.loc[index, 'degree'] = M.loc[index, 'degree']
                            temp_block_M.loc[index, 'destination_nodes'] = np.array(per_node)
                        else:
                            temp_block_M.loc[index, 'destination_nodes'] = np.append(
                                temp_block_M.loc[index, 'destination_nodes'], per_node)
            M_block_stripe.append(temp_block_M)
            bar.update(1)
    return M_block_stripe


# print(M_block_stripe)

# 计算每个节点的入度，暂时没有用上

def comput_node_input_time(nodes):
    node_input_time = nodes.apply(pd.value_counts)['output_node']
    return node_input_time

# 计算每个节点的出度，暂时没有用上


def comput_node_output_time(nodes):
    node_output_time = nodes.apply(pd.value_counts)['input_node']
    return node_output_time

# quick block_strip algorithm


def quick_block_strip(nodes, block_node_groups,node_output_time):
    # 存最后的各个划分后的M
    M_block_stripe = []
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            temp_block_M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
            temp_block_M.set_index('source_node', inplace=True)
            # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
            for per_node in node_group:
                # for index,row in nodes.iterrows()
                nodes.set_index('input_node', inplace = True)
                output_node = nodes.loc[per_node,'output_node']
                if per_node not in temp_block_M.index.tolist():
                    temp_block_M.loc[per_node, 'degree'] = node_output_time[per_node]
                    temp_block_M.loc[per_node, 'destination_nodes'] = np.array(output_node)
                else:
                    temp_block_M.loc[per_node, 'destination_nodes'] = np.append(temp_block_M.loc[per_node, 'destination_nodes'], output_node)
            M_block_stripe.append(temp_block_M)
            bar.update(1)
    return M_block_stripe
# 计算pagerank值


def pageRank(block_stripe_M, old_rank,all_node):
    num = len(all_node)
    initial_rank_new = (1-Beta)/ num
    new_rank = pd.DataFrame({'page': all_node, 'score': initial_rank_new}, columns=['page', 'score'])
    new_rank.set_index('page',inplace=True)
    sum_new_sub_old = 0
    for index, row in old_rank.iterrows():
        sum_new_sub_old += math.fabs(new_rank.loc[index, 'score'] - old_rank.loc[index, 'score'])
    while sum_new_sub_old < derta:
        for per_M in block_stripe_M:
            for index, row in per_M.iterrows():
                node_list = row['destination_nodes'].tolist()
                for per_node in node_list:
                    new_rank.loc[per_node,'score'] += Beta*old_rank.loc[index,'score']/row['degree']
        # 解决dead-ends和Spider-traps
        # 所有new_rank的score加和得s，再将每一个new_rank的score加上(1-sum)/len(all_node)，使和为1
        # s = 0
        s = sum(new_rank['score'].values)
        ss = (1-s) / num
        # 这个需要再看看
        new_rank['score'].apply(lambda x: x+ss)
        # for index, row in new_rank:
        #     new_rank.loc[index, 'score'] += ss
        old_rank = new_rank
    return new_rank


# 相当于main，输入文件路径，输出rank值
def mypageRank(file):
    nodes, all_node = load_data(file,output_csv = False)
    # rank = pd.DataFrame(columns=['page', 'score'])
    # # 将page列设置为索引
    # rank.set_index('page', inplace=True)
    rank = generate_rank(all_node)
    # print(rank)
    # M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
    # 将M的source_node列设置为索引
    # M.set_index('source_node', inplace=True)
    M = nodes_to_M(nodes)

    step = 100
    block_node_groups = list_to_groups(all_node, step)
    M_block_stripe = block_strip(M, block_node_groups)

    # new_rank = pd.DataFrame(columns=['page', 'score'])
    # new_rank.set_index('page', inplace=True)
    new_rank = pageRank(M_block_stripe, rank, all_node)
    return new_rank

# 线程相关
class myThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.counter = counter

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        # threadLock.acquire()
        file = 'WikiData2.txt'
        new_rank = mypageRank(file)
        print(new_rank)

    # 释放锁，开启下一个线程
    # threadLock.release()


if __name__ == '__main__':
    threadLock = threading.Lock()
    threads = []
    # 创建新线程
    thread1 = myThread(1, "Thread-1")
    # thread2 = myThread(2, "Thread-2", 2)
    # 开启新线程
    thread1.start()
    # thread2.start()
    thread1.join()
    print("退出主线程")
    # @timeit(mypageRank('WikiData2.txt'))
    # print(new_rank)