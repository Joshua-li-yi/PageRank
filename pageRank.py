import time

import networkx.classes.digraph
import pandas as pd
from pandarallel import pandarallel
import matplotlib.pyplot as plt
import numpy as np
import graphviz
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool
# from bokeh.models.graphs import from_networkx
# from bokeh.transform import cumsum
# from bokeh.palettes import Category20c,Spectral4
import streamlit as st
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\graphviz-2.38\release\bin' # 这里是 path+ Graphviz/bin 即 Graphviz 的 bin目录
pandarallel.initialize(nb_workers=4)
# 设置参数
Beta = 0.85
derta = 0.0001
all_line = 103690
# 设置取的随机行数的比例
row_frac = 0.001

# 设置迭代动图的参数
x = [0]
y = [1.0]
# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


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
    initial_old_rank = 1 / len(all_node)
    rank = pd.DataFrame({'page': all_node, 'score': initial_old_rank}, columns=['page', 'score'])
    # 将page列设置为索引
    rank.set_index('page', inplace=True)
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


# 新的分块方法，原先使用dataframe格式存的分块
# 现在改为使用list格式，相应读取时也要使用list格式的方法


def quick_block_stripe(nodes, block_node_groups):
    # 存最后的各个划分后的M
    node_output_time = comput_node_output_time(nodes)
    M_block_list = []
    # 根据input_node 进行分组进行分组
    grouped = nodes.groupby('input_node')
    # print(grouped)
    # with tqdm(total=len(block_node_groups), desc='block strip progress') as bar:
    #     for node_group in block_node_groups:
    #         # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
    #         for key, group in grouped:
    #             # print(group)
    #             output_node_list = group['output_node'].values.tolist()
    #             intersect_set = set(node_group).intersection(output_node_list)
    #             intersect_set = list(intersect_set)
    #             # np.where(len(intersect_set),M_block_list.append([]))
    #             if len(intersect_set):
    #                 M_block_list.append([key, node_output_time[key], intersect_set])
    #         bar.update(1)
    temp_len = len(block_node_groups)
    st.info("block strip progress")
    bar = st.progress(0)
    temp_i = 0
    for node_group in block_node_groups:
        temp_i += 1
        # 将大的M 根据 划分后的node节点，进行块条化最后结果存到M_block_stripe列表中
        for key, group in grouped:
            # print(group)
            output_node_list = group['output_node'].values.tolist()
            intersect_set = set(node_group).intersection(output_node_list)
            intersect_set = list(intersect_set)
            # np.where(len(intersect_set),M_block_list.append([]))
            if len(intersect_set):
                M_block_list.append([key, node_output_time[key], intersect_set])
        bar.progress(temp_i/temp_len)
    return M_block_list


def pageRank(M_list, old_rank, all_node):
    num = len(all_node)
    initial_rank_new = (1 - Beta) / num
    sum_new_sub_old = 1.0
    new_rank = pd.DataFrame({'page': all_node}, columns=['page', 'score'])
    new_rank.set_index('page', inplace=True)
    while sum_new_sub_old > derta:
        new_rank['score'] = initial_rank_new

        for m in M_list:
            # print(m)
            temp_old_rank = old_rank.loc[m[0], 'score']

            temp_degree = m[1]
            for per_node in m[2]:
                new_rank.loc[per_node, 'score'] += Beta * temp_old_rank / temp_degree

        # 解决dead-ends和Spider-traps
        # 所有new_rank的score加和得s，再将每一个new_rank的score加上(1-sum)/len(all_node)，使和为1
        s = new_rank['score'].values.sum()
        ss = (1 - s) / num
        new_rank['score'] += ss

        # 计算sum_new_sub_old

        old_rank['score'] = new_rank['score'] - old_rank['score']
        old_rank['score'] = old_rank['score'].abs()
        sum_new_sub_old = np.sum(old_rank['score'].values)

        old_rank['score'] = new_rank['score']

    print('rank compute finish')
    return new_rank


# 相当于main，输入文件路径，输出rank值
# step 设置块条化的步长
def mypageRank(nodes, all_node, step):
    # nodes, all_node = load_data(file, output_csv=False, frac=row_frac)
    # global new_rank
    rank = generate_rank(all_node)
    pre_process(nodes)
    # print(rank)
    # 将allnode分成小块
    block_node_groups = list_to_groups(all_node, step)
    # print(block_node_groups)
    # quick block strip
    start_quick_block = time.clock()
    # M_block_stripe = quick_block_stripe(nodes,block_node_groups)
    M_block_list = quick_block_stripe(nodes, block_node_groups)
    # print(M_block_stripe)
    # print(M_block_list)
    end_quick_block = time.clock()
    print('Running time: %s Seconds' % (end_quick_block - start_quick_block))
    # print(M_block_stripe)
    # 计算pagerank值
    start_pagerank = time.clock()
    # new_rank = pageRank(M_block_stripe, rank, all_node)
    new_rank = pageRank(M_block_list, rank, all_node)
    end_pagerank = time.clock()
    print('Running time: %s Seconds' % (end_pagerank - start_pagerank))
    new_rank.sort_values('score', inplace=True, ascending=0)
    sort_rank = new_rank.head(100)
    return sort_rank

st.title("Show the resoult of pageRank")
st.write("parameter control")
Beta = st.slider(label='teleport', min_value=0., max_value=1.,key=1)
row_frac = st.slider(label='frac', min_value=0., max_value=1.,key=2)
st.write("when parameter are ","teleport=",Beta,"frac = ",row_frac, "the result is below")
nodes = pd.DataFrame()
all_node = []
# btn_import_data = st.button("import url data")
upload_file = st.file_uploader("Chooose a txt file data", type="txt")
if upload_file is not None:
    temp_nodes, temp_all_node = load_data(upload_file, frac=row_frac)
    nodes = temp_nodes
    all_node = temp_all_node
    st.success('import data success')
    # st.write(nodes)
    # st.write(all_node)

def comput_rank():
    # temp_step = np.floor(len(all_node)/5)
    # np.floor()
    temp_scores = mypageRank(nodes, all_node, step=100)
    # 将page一列重新转化为非index列，并增加新的一列
    temp_scores = temp_scores.reset_index()
    # 从1开始索引
    temp_scores.index += 1
    return temp_scores


btn_compute_pageRank = st.button("compute pageRank")
if btn_compute_pageRank:
    scores = comput_rank()
    st.info("the page and score are below")
    st.table(scores)
    st.success("compute pageRank success")

btn_show_pageRank = st.button("show pageRank result chart")
if btn_show_pageRank:
    scores = comput_rank()
    x = scores['page'].tolist()
    y = scores['score'].tolist()

    st.info("network relation graph")
    graph = graphviz.Digraph()
    # graph
    nodes.apply(lambda row: graph.node(str(row[0])))
    nodes.apply(lambda row: graph.node(str(row[1])))
    # nodes.apply(lambda row: st.write(str(row[0]),str(row[1])),axis=1)
    nodes.apply(lambda row: graph.edge(str(row[0]), str(row[1])), axis=1)
    # graph.edge(nodes['input_node'].tolist(), nodes[].tolist())
    # graphviz.render("a")
    # st.pyplot()
    st.graphviz_chart(graph)
    st.info("bar chart")
    plt.bar(x, y)
    plt.ylabel("score")
    plt.xlabel("page")
    st.pyplot()

    st.info("line chart")
    plt.plot(x, y)
    plt.ylabel("score")
    plt.xlabel("page")
    st.pyplot()

    st.info("box chart")
    scores.set_index('page', inplace=True)
    scores.boxplot()
    st.pyplot()



if st.button("Celebrate"):
    st.balloons()


