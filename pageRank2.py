import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

# 设置参数
Beta = 0.85
derta = 0.00001
# 设置取的随机行数的比例
row_frac = 1.

# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 将结果写入文件
def writeResult(new_rank):
    file_path = "result.txt"
    new_rank = new_rank.reset_index()
    with open(file_path, "w") as f:
        new_rank.apply(lambda row: f.write('['+str(int(row[0]))+'] ['+str(row[1])+']\n'), axis=1)
    print('result data write finish')


# 从txt导入数据、将数据转化成 csv 格式 nodes 输入和输出，类似于将边存起来
# 输出nodes--------dataframe 格式 和all_node --------list,frac 设置随机取文件中数的比例
def load_data(filePath, output_csv=False, frac=1.):
    print('begin load data')
    txt = np.loadtxt(filePath)
    nodes = pd.DataFrame(data=txt, columns=['input_node', 'output_node'])
    # 将值转化为int类型
    nodes['input_node'] = nodes['input_node'].astype(int)
    nodes['output_node'] = nodes['output_node'].astype(int)
    # 设置随机取多少行
    if frac != 1.:
        print('random select', frac * 100, '% data')
        nodes = nodes.sample(frac=frac, random_state=1)
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
    print('load data finish')
    return nodes, all_node


# 预处理函数
def pre_process(nodes):
    print('begin Preprocessing')
    print('Determine whether there are duplicate lines')
    print(nodes[nodes.duplicated()])
    print('Preprocessing finish')


# 生成rank值
def generate_rank(all_node):
    # 初始rank
    initial_old_rank = 1 / len(all_node)
    rank = {node:initial_old_rank for node in all_node}
    print('generate initial rank finish')
    return rank


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


# 计算每个节点的出度
def comput_node_output_time(nodes):
    node_output_time = nodes.apply(pd.value_counts)['input_node']
    return node_output_time


# 块条化
def block_stripe(nodes, block_node_groups):
    # 存最后的各个划分后的M
    node_output_time = comput_node_output_time(nodes)
    # print(node_output_time[6])
    M_block_list = []
    # 根据input_node 进行分组进行分组
    grouped = nodes.groupby('input_node')
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
            for key, group in grouped:
                output_node_list = group['output_node'].values.tolist()
                intersect_set = set(node_group).intersection(output_node_list)
                intersect_set = list(intersect_set)
                if len(intersect_set):
                    M_block_list.append([key, node_output_time[key], intersect_set])
            bar.update(1)
    return M_block_list


def pageRank(M_list, old_rank, all_node):
    num = len(all_node)
    initial_rank_new = (1 - Beta) / num
    sum_new_sub_old = 1.0

    i = int(1)
    print("start the iteration")
    while sum_new_sub_old > derta:
        new_rank = {node: initial_rank_new for node in all_node}
        for m in M_list:
            temp_old_rank = old_rank[m[0]]
            temp_degree = m[1]
            for per_node in m[2]:
                new_rank[per_node] += Beta * temp_old_rank / temp_degree
        # 解决dead-ends和Spider-traps
        # 所有new_rank的score加和得s，再将每一个new_rank的score加上(1-sum)/len(all_node)，使和为1
        s = sum(new_rank.values())
        ss = (1 - s) / num
        new_rank = {k: new_rank[k]+ss for k in new_rank}

        # 计算sum_new_sub_old
        temp_list = list(map(lambda x:abs(x[0] - x[1]), zip(new_rank.values(), old_rank.values())))
        sum_new_sub_old = np.sum(temp_list)

        old_rank = new_rank
        print('iterations:', i, 'sum_new_sub_old:', sum_new_sub_old)
        i += 1
    print('rank compute finish')
    return old_rank


# 相当于main，输入文件路径，输出rank值
# step 设置块条化的步长
def mypageRank(file, step):
    nodes, all_node = load_data(file, output_csv=False, frac=row_frac)
    rank = generate_rank(all_node)
    pre_process(nodes)
    # 将allnode分成小块
    block_node_groups = list_to_groups(all_node, step)

    # quick block strip
    start_quick_block = time.perf_counter()
    M_block_list = block_stripe(nodes, block_node_groups)
    end_quick_block = time.perf_counter()
    print('Running time: %s Seconds' % (end_quick_block - start_quick_block))
    # print(M_block_stripe)
    # 计算pagerank值
    start_pagerank = time.perf_counter()
    new_rank = pageRank(M_block_list, rank, all_node)
    end_pagerank = time.perf_counter()
    print('Running time: %s Seconds' % (end_pagerank - start_pagerank))
    # 转化为df类型
    new_rank = pd.DataFrame(new_rank.items(), columns=['page','score'])
    new_rank.set_index('page',inplace=True)

    # rank排序 从大到小
    new_rank.sort_values('score', inplace=True, ascending=0)
    # 取前一百
    sort_rank = new_rank.head(100)
    # print(sort_rank)
    return sort_rank


if __name__ == '__main__':
    start_main = time.perf_counter()
    # 文件位置
    file = 'WikiData.txt'
    # 开始计算
    new_rank = mypageRank(file, step=10000)
    # 写入数据
    writeResult(new_rank)
    end_main = time.perf_counter()
    print('Running time: %s Seconds' % (end_main - start_main))
    print('process end')
    # os.system('pause')
