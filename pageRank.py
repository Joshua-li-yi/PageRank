from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import threading

# 设置参数
Beta = 0.85
derta = 0.0001
all_line = 103690
# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def load_data(filePath):
    nodes = pd.DataFrame(columns=['input_node', 'output_node'])
    all_node = []
    data_file = open(filePath, 'r')
    line = data_file.readline()
    line_num = 1
    with tqdm(total=all_line,desc='data load progress') as bar:
        while line:
            temp_input, temp_output = line.split('\t')
            temp_output = temp_output[:-1]
            # print(type(temp_input))
            temp_input = int(temp_input)
            temp_output = int(temp_output)

            nodes.loc[line_num] = [temp_input,temp_output]

            if temp_input not in all_node:
                all_node.append(temp_input)
            if temp_output not in all_node:
                all_node.append(temp_output)
            # print(line_num)
            line_num += 1
            line = data_file.readline()
            bar.update(1)
    # print('line all')
    # print(line_num)
    data_file.close()
    # 根据inputpage的值排序
    nodes.sort_values('input_node',inplace=True)
    # 重置索引
    nodes.reset_index(inplace=True, drop=True)
    # all_note 升序排列
    all_node.sort()
    return nodes ,all_node

def generate_rank(all_node):
    # 初始rank
    initial_rank_old = 1/len(all_node)
    rank = pd.DataFrame({'page':all_node,'score':initial_rank_old},columns=['page','score'])
    # 将page列设置为索引

    tqdm.pandas(desc="rank initial")
    rank.progress_apply(lambda x: x ** 2)
    rank.set_index('page',inplace=True)
    return rank


def nodes_to_M(nodes):
    M = pd.DataFrame(columns=['source_node','degree','destination_nodes'])
    # 将M的source_node列设置为索引
    M.set_index('source_node', inplace=True)
    with tqdm(total=nodes.shape[0], desc='M matrix generate progress') as bar:
        for index, node_row in nodes.iterrows():
            tmp_list = M.index.tolist()
            if node_row[0] not in tmp_list:
                M.loc[node_row[0], 'degree'] = int(0)
                M.loc[node_row[0], 'destination_nodes'] = np.array([])
            else:
                M.loc[node_row[0],'degree'] += 1
                M.loc[node_row[0],'destination_nodes'] = np.append(M.loc[node_row[0],'destination_nodes'],node_row[1])
            bar.update(1)
    return M


def list_to_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) * per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list


# block_strip algorithm


def block_strip(M,block_node_groups):
    M_block_stripe = []
    with tqdm(total=len(block_node_groups),desc='block strip progress') as bar:
        for node_group in block_node_groups:
            temp_block_M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
            temp_block_M.set_index('source_node', inplace=True)
            for per_node in node_group:
                for index,row in M.iterrows():
                    if per_node in row['destination_nodes'].tolist():
                        tmp_list = temp_block_M.index.tolist()
                        if per_node not in tmp_list:
                            temp_block_M.loc[index,'degree'] = M.loc[index,'degree']
                            temp_block_M.loc[index,'destination_nodes'] = np.array(per_node)
                        else:
                            temp_block_M.loc[index, 'destination_nodes'] = np.append(
                                temp_block_M.loc[index, 'destination_nodes'], per_node)
            M_block_stripe.append(temp_block_M)
            bar.update(1)
    return M_block_stripe
# print(M_block_stripe)


def comput_node_input_time(nodes):
    node_input_time = nodes.apply(pd.value_counts)['output_node']
    return node_input_time


def pageRank(block_stripe_M, old_rank,all_node):
    initial_rank_new = (1-Beta)/ len(all_node)
    new_rank = pd.DataFrame({'page': all_node, 'score': initial_rank_new}, columns=['page', 'score'])
    new_rank.set_index('page',inplace=True)
    sum_new_sub_old = 0
    for index, row in old_rank.iterrows():
        sum_new_sub_old += math.fabs(new_rank.loc[index, 'score'] - old_rank.loc[index, 'score'])
    while sum_new_sub_old < derta:
        for per_M in block_stripe_M:
            for index, row in per_M:
                node_list = row['destination_nodes'].tolist()
                for per_node in node_list:
                    new_rank.loc[per_node,'score'] += Beta*old_rank.loc[index,'score']/row['degree']
    return new_rank

def mypageRank(file):
    file_path = file
    nodes, all_node = load_data(file_path)

    rank = pd.DataFrame(columns=['page', 'score'])
    # 将page列设置为索引
    rank.set_index('page', inplace=True)
    rank = generate_rank(all_node)
    print(rank)
    M = pd.DataFrame(columns=['source_node', 'degree', 'destination_nodes'])
    # 将M的source_node列设置为索引
    M.set_index('source_node', inplace=True)
    M = nodes_to_M(nodes)

    step = 100
    block_node_groups = list_to_groups(all_node, step)
    M_block_stripe = block_strip(M, block_node_groups)

    new_rank = pd.DataFrame(columns=['page', 'score'])
    new_rank.set_index('page', inplace=True)
    new_rank = pageRank(M_block_stripe, rank,all_node)
    return new_rank


class myThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.counter = counter

    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        # threadLock.acquire()
        file = 'WikiData.txt'
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
    # print(new_rank)
