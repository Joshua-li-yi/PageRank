import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from numba import jit
from pandarallel import pandarallel
import dask.dataframe as dd

pandarallel.initialize(nb_workers=4)
# 设置参数
Beta = 0.85
derta = 0.0001
all_line = 103690
# 设置取的随机行数的比例
row_frac = 1
# 设置迭代动图的参数
x = [0]
y = [1.0]
# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def writeResult(new_rank):
    file_path = "result.txt"
    with open(file_path, "w") as f:
        for index, row in new_rank.iterrows():
            f.write("[")
            f.write(str(index))
            f.write("] ")
            f.write("[")
            f.write(str(new_rank.loc[index, 'score']))
            f.write("]\n")
    print('result data write finish')


# 从txt导入数据、将数据转化成 csv 格式 nodes 输入和输出，类似于将边存起来
# 输出nodes--------dataframe 格式 和all_node --------list,frac 设置随机取文件中数的比例
def load_data(filePath, output_csv=False,frac = 1.):
    print('begin load data')
    txt = np.loadtxt(filePath)
    nodes = pd.DataFrame(data=txt,columns=['input_node', 'output_node'])
    # 将值转化为int类型
    nodes['input_node'] = nodes['input_node'].astype(int)
    nodes['output_node'] = nodes['output_node'].astype(int)
    # 设置随机取多少行
    if frac != 1.:
        print('random select',frac*100,'% data')
        nodes = nodes.sample(frac=frac,random_state=1)
    if output_csv is True:
        nodes.to_csv('WikiData2.csv')
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
    initial_old_rank =1/len(all_node)
    rank = pd.DataFrame({'page': all_node, 'score': initial_old_rank}, columns=['page', 'score'])
    # 将page列设置为索引
    rank.set_index('page', inplace=True)
    print('generate initial rank finish')
    return rank


# 将之前得nodes 存起来得边，转化为矩阵。用的是老师PPT上的'source_node','degree','destination_nodes'结构
global M
M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
# 将M的source_node列设置为索引
M.set_index('source_node', inplace=True)


def to_M(node_row):
    input_node = node_row[0]
    output_node = node_row[1]
    global M
    if input_node not in M.index.tolist():
        M.loc[input_node, 'degree'] = int(1)
        M.loc[input_node, 'destination_nodes'] = np.array([output_node])
    else:
        M.loc[input_node, 'degree'] += 1
        M.loc[input_node, 'destination_nodes'] = np.append(M.loc[input_node, 'destination_nodes'], output_node)


# 加上这个快了一点
@jit(forceobj=True)
def nodes_to_M(nodes):
    # 这种方式比较快
    nodes.apply(to_M, axis=1)
    # with tqdm(total=nodes.shape[0], desc='M matrix generate progress') as bar:
    #     for index, node_row in nodes.iterrows():
    #         input_node = node_row[0]
    #         output_node = node_row[1]
    #         if input_node not in M.index.tolist():
    #             M.loc[input_node, 'degree'] = int(1)
    #             M.loc[input_node, 'destination_nodes'] = np.array([output_node])
    #         else:
    #             M.loc[input_node, 'degree'] += 1
    #             M.loc[input_node, 'destination_nodes'] = np.append(M.loc[input_node, 'destination_nodes'], output_node)
    #         bar.update(1)
    print('generate M finish')
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
# 超级超级慢，一个多小时，要么看看怎么改进，要么修改下面的quick_block_strip函数

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
                        if index not in temp_block_M.index.tolist():
                            temp_block_M.loc[index, 'degree'] = row['degree']
                            temp_block_M.loc[index, 'destination_nodes'] = np.array(per_node)
                        else:
                            temp_block_M.loc[index, 'destination_nodes'] = np.hstack((temp_block_M.loc[index, 'destination_nodes'],per_node))

            M_block_stripe.append(temp_block_M)

            bar.update(1)
    print('block strip finish')
    return M_block_stripe


# 改进block_stripe算法
def block_stripe2(M,block_node_groups):
    M_block_stripe = []
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            temp_block_M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
            temp_block_M.set_index('source_node', inplace=True)
            for index,row in M.iterrows():
                intersect_set = set(node_group).intersection(row['destination_nodes'].tolist())
                intersect_set = list(intersect_set)
                if len(intersect_set) != 0:
                    temp_block_M.loc[index, 'degree'] = row['degree']
                    temp_block_M.loc[index,'destination_nodes'] = np.array(intersect_set)
                # for destination_node in row['destination_nodes'].tolist():
                #     if destination_node in node_group:
                #         if index not in temp_block_M.index.tolist():
                #
                #             temp_block_M.loc[index, 'destination_nodes'] = np.array(destination_node)
                #         else:
                #             temp_block_M.loc[index, 'destination_nodes'] = np.hstack(
                #                 (temp_block_M.loc[index, 'destination_nodes'], destination_node))
            M_block_stripe.append(temp_block_M)
            bar.update(1)
    print('block strip finish')
    return M_block_stripe


# 计算每个节点的入度，暂时没有用上

def comput_node_input_time(nodes):
    node_input_time = nodes.apply(pd.value_counts)['output_node']
    return node_input_time

# 计算每个节点的出度


def comput_node_output_time(nodes):
    node_output_time = nodes.apply(pd.value_counts)['input_node']
    return node_output_time

# quick block_strip algorithm
# 输入nodes，直接分块，不用转M


def quick_block_stripe(nodes, block_node_groups):
    # 存最后的各个划分后的M
    node_output_time = comput_node_output_time(nodes)
    # print(node_output_time[6])
    M_block_stripe = []
    # 根据input_node 进行分组进行分组
    grouped = nodes.groupby('input_node')
    # print(grouped)
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            temp_block_M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
            temp_block_M.set_index('source_node', inplace=True)
            # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
            for key,group in grouped:
                # print(group)
                output_node_list = group['output_node'].values.tolist()
                intersect_set = set(node_group).intersection(output_node_list)
                intersect_set = list(intersect_set)

                if len(intersect_set) != 0:
                    temp_block_M.loc[key, 'degree'] = node_output_time[key]
                    temp_block_M.loc[key, 'destination_nodes'] = np.array(intersect_set)
            M_block_stripe.append(temp_block_M)
            bar.update(1)
    return M_block_stripe

# 新的分块方法，原先使用dataframe格式存的分块
# 现在改为使用list格式，相应读取时也要使用list格式的方法


# @jit(forceobj=True)
def quick_block_stripe2(nodes, block_node_groups):
    # 存最后的各个划分后的M
    node_output_time = comput_node_output_time(nodes)
    # print(node_output_time[6])
    M_block_list = []
    # 根据input_node 进行分组进行分组
    # dd_nodes = dd.from_pandas(nodes,npartitions=10)
    grouped = nodes.groupby('input_node')
    # dd_grouped = dd_nodes.groupby('input_node')
    # print(grouped)
    with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
        for node_group in block_node_groups:
            # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
            for key,group in grouped:
                # print(group)
                output_node_list = group['output_node'].values.tolist()
                intersect_set = set(node_group).intersection(output_node_list)
                intersect_set = list(intersect_set)
                # np.where(len(intersect_set),M_block_list.append([]))
                if len(intersect_set):
                    M_block_list.append([key, node_output_time[key], intersect_set])
            bar.update(1)
    return M_block_list
# 计算pagerank值


def update_rank(row,oldrank):
    node_list = row['destination_nodes'].tolist()
    if isinstance(node_list, int):
        node_list = [node_list]
    global new_rank
    for per_node in node_list:
        new_rank.loc[per_node, 'score'] += Beta * oldrank.loc[row.name, 'score'] / row['degree']
    return row


def pageRank(block_stripe_M, old_rank,all_node):
    num = len(all_node)
    initial_rank_new = (1-Beta)/ num

    new_rank = pd.DataFrame({'page': all_node}, columns=['page', 'score'])
    new_rank.set_index('page', inplace=True)
    sum_new_sub_old = 1.0
    # iteration_time = 0

    while sum_new_sub_old > derta:
        new_rank['score'] = initial_rank_new
        # iteration_time += 1
        # x.append(a)

        for per_M in block_stripe_M:
            # 此处可以改进
            # 发现丫的还不如for循环快
            # per_M.apply(update_rank,axis=1,args=(old_rank,))
            for index, row in per_M.iterrows():
                node_list = row['destination_nodes'].tolist()
                # 如果满足nodelist 则node_list = [node_list]
                if isinstance(node_list,int):
                    node_list = [node_list]
                np.where(isinstance(node_list,int),node_list = [node_list])
                # 此处可以加速改进
                # new_rank = new_rank.apply(lambda k: update_rank(row=k, old_score=tmp_value, degree=degree) if k.index in node_list else k, axis=1)
                temp_old_rank = old_rank.loc[index, 'score']
                temp_degree = row['degree']
                for per_node in node_list:
                    new_rank.loc[per_node, 'score'] += Beta * temp_old_rank / temp_degree

        # 解决dead-ends和Spider-traps
        # 所有new_rank的score加和得s，再将每一个new_rank的score加上(1-sum)/len(all_node)，使和为1
        # s = 0
        s = sum(new_rank['score'].values)
        ss = (1-s) / num
        new_rank['score'] += ss
        # 计算sum_new_sub_old
        tmp = old_rank
        tmp['score'] = new_rank['score']-old_rank['score']
        tmp['score'] = tmp['score'].abs()
        sum_new_sub_old = sum(tmp['score'].values)

        # print(sum_new_sub_old)
        old_rank['score'] =new_rank['score']

        # 绘制迭代动图
        # 未完成
        # y.append(sum_new_sub_old)
        # ani = animation.FuncAnimation(fig=fig,
        #                               func=update(sum_new_sub_old),
        #                               frames=1,
        #                               init_func=init,
        #                               interval=20,
        #                               blit=False)
        # plt.show()
    print('rank compute finish')
    return new_rank

# @jit(cache=False,  nogil=True, parallel=True)
# def speedup_sum(z):
#     return np.sum(z)

# @jit(cache=False, nogil=True, parallel=True)
# def speedup_sum2(z, s):
#     return z+s

def pageRank2(M_list, old_rank,all_node):
    num = len(all_node)
    initial_rank_new = (1-Beta)/ num
    sum_new_sub_old = 1.0
    new_rank = pd.DataFrame({'page': all_node}, columns=['page', 'score'])
    new_rank.set_index('page', inplace=True)
    # iteration_time = 0
    while sum_new_sub_old > derta:
        new_rank['score'] = initial_rank_new
        # iteration_time += 1
        # x.append(a)
        # temp_old_rank_list = []
        for m in M_list:
            # print(m)
            temp_old_rank = old_rank.loc[m[0], 'score']
            # temp_old_rank_list.append(temp_old_rank)
            temp_degree = m[1]
            for per_node in m[2]:
                new_rank.loc[per_node, 'score'] += Beta * temp_old_rank / temp_degree
                # new_rank.loc[per_node, 'score'].compute()
        # 解决dead-ends和Spider-traps
        # 所有new_rank的score加和得s，再将每一个new_rank的score加上(1-sum)/len(all_node)，使和为1
        # s = 0
        #
        # dd_new_rank = dd.from_pandas(new_rank, npartitions=40)
        # dd_old_rank = dd.from_pandas(old_rank, npartitions=40)
        # s = dd_new_rank['score'].values.sum().compute()
        s = new_rank['score'].values.sum()
        ss = (1-s) / num
        # dd_new_rank.score = dd_new_rank.score+ss
        # dd_new_rank.score.compute()
        new_rank['score'] += ss

        # 计算sum_new_sub_old

        old_rank['score'] = new_rank['score']-old_rank['score']
        old_rank['score'] = old_rank['score'].abs()
        sum_new_sub_old = np.sum(old_rank['score'].values)

        # tmp = dd_new_rank.score - dd_old_rank.score
        # tmp.compute()
        # tmp2 = tmp.abs()
        # tmp2.compute()
        # sum_new_sub_old = tmp2.values.sum().compute()

        # print(sum_new_sub_old)
        # temp_list = list(dd_new_rank['score'])
        # tmp_df = pd.DataFrame({'page':all_node,'score':temp_list},columns=['page','score'])
        # new_rank = tmp_df
        # new_rank.set_index('page',inplace=True)

        old_rank['score'] = new_rank['score']
        # print(old_rank)
        # 绘制迭代动图
        # 未完成
        # y.append(sum_new_sub_old)
        # ani = animation.FuncAnimation(fig=fig,
        #                               func=update(sum_new_sub_old),
        #                               frames=1,
        #                               init_func=init,
        #                               interval=20,
        #                               blit=False)
        # plt.show()

    print('rank compute finish')
    return new_rank

# 相当于main，输入文件路径，输出rank值
# step 设置块条化的步长
def mypageRank(file,step):
    nodes, all_node = load_data(file,output_csv=False,frac=row_frac)
    # global new_rank

    rank = generate_rank(all_node)
    pre_process(nodes)
    # print(rank)

    # slow block stripe
    # start_M = time.clock()
    # M = nodes_to_M(nodes)
    # end_M = time.clock()
    # print('Running time: %s Seconds' % (end_M - start_M))
    # print(M)

    # 将allnode分成小块
    block_node_groups = list_to_groups(all_node, step)
    # print(block_node_groups)
    # M_block_stripe = block_strip(M, block_node_groups)
    # M_block_stripe = block_stripe2(M, block_node_groups)

    # quick block strip
    start_quick_block = time.clock()
    # M_block_stripe = quick_block_stripe(nodes,block_node_groups)
    M_block_list = quick_block_stripe2(nodes,block_node_groups)
    # print(M_block_stripe)
    # print(M_block_list)
    end_quick_block = time.clock()
    print('Running time: %s Seconds' % (end_quick_block - start_quick_block))
    # print(M_block_stripe)
    # 计算pagerank值
    start_pagerank = time.clock()
    # new_rank = pageRank(M_block_stripe, rank, all_node)
    new_rank = pageRank2(M_block_list, rank, all_node)
    end_pagerank = time.clock()
    print('Running time: %s Seconds' % (end_pagerank - start_pagerank))
    return new_rank


if __name__ == '__main__':
    start_main = time.clock()
    # 文件位置
    file = 'WikiData.txt'
    # 开始计算
    new_rank = mypageRank(file,step=100)
    print(new_rank)
    # rank排序
    new_rank.sort_values('score',inplace=True,ascending=0)
    sort_rank = new_rank.head(100)
    # 写入数据
    writeResult(sort_rank)
    end_main = time.clock()
    print('Running time: %s Seconds' % (end_main - start_main))
