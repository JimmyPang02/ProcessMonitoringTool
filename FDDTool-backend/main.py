import os
import csv
import numpy as np
import matlab.engine
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.io as pio
from fastapi.responses import HTMLResponse
import subprocess
import copy
from DiPCA import runDiPCA
from SFA import runSFA
from DiPLS import runDiPLS

# Matlab引擎
eng = matlab.engine.start_matlab()

# FastAPI实例
app = FastAPI()
# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以设置为适当的前端域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储上传的文件
uploaded_files_path = Path("uploaded_files")
uploaded_files_path.mkdir(exist_ok=True)
# 存储可视化结果
plot_results_path = Path("plot_results")
plot_results_path.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), type: str = "normal"):
    if type not in ["normal", "fault"]:
        return JSONResponse(content={"message": "Invalid data type"}, status_code=400)
    file_path = uploaded_files_path / type / file.filename
    with file_path.open("wb") as f:
        f.write(file.file.read())
    return {"message": "success"}


@app.get("/DataList")
async def get_data_list(type: str = "normal"):
    if type not in ["normal", "fault"]:
        return JSONResponse(content={"message": "Invalid data type"}, status_code=400)

    files_list = os.listdir(uploaded_files_path / type)

    return {"data": files_list, "message": "success"}


@app.get("/delete")
async def delete_data(file_name: str, type: str = "normal"):
    if type not in ["normal", "fault"]:
        return JSONResponse(content={"message": "Invalid data type"}, status_code=400)

    file_path = uploaded_files_path / type / file_name
    if file_path.is_file():
        os.remove(file_path)
        return {"message": "success"}
    else:
        return JSONResponse(content={"message": "File not found"}, status_code=404)


class RunRequest(BaseModel):
    normal_name: str
    fault_name: str
    isReturnMetric: bool
    method: str


def plot_cva(T2mon, Qmon, Ta, Qa, fault_labels):
    """CVA模型的结果可视化

    Args:
        T2mon (list): T2统计量
        Qmon (list): Q统计量
        Ta (float): T2阈值
        Qa (float): Q阈值
        fault_labels (list): 故障数据标签列

    Returns:
        html_file_name (str): 可视化结果的html文件名
    """

    N = len(T2mon)  # 获取数据长度
    fig = go.Figure()

    # 使用make_subplots创建包含三个子图的图表
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'T^2', 'SPE', 'Fault Labels'))

    # 添加第一个子图,显示T2mon
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=T2mon,
                             mode='lines', name='T2mon'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        Ta, Ta], mode='lines', name='Ta', line=dict(dash='dash')), row=1, col=1)

    # 添加第二个子图,显示Qmon
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=Qmon,
                             mode='lines', name='Qmon'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        Qa, Qa], mode='lines', name='Qa', line=dict(dash='dash')), row=2, col=1)

    # 添加第三个子图，显示故障数据标签列
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=fault_labels,
                             mode='markers+lines', name='Fault Labels',
                             marker={"size": 4, }), row=3, col=1)  # 点的大小

    # 更新布局
    fig.update_yaxes(type='log', title='T^2',
                     row=1, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=1, col=1, automargin=True)
    fig.update_yaxes(type='log', title='SPE',
                     row=2, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=2, col=1, automargin=True)
    fig.update_yaxes(title='Fault Labels', row=3, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=3, col=1, automargin=True)
    fig.update_layout(height=1000, width=1000,
                      legend=dict(
                          yanchor="bottom",
                          y=1.07,
                          xanchor="center",
                          x=0.5,
                          bgcolor='rgba(0,0,0,0)',  # 设置边框颜色为透明
                          borderwidth=0,  # 设置边框宽度为0，即隐藏边框
                          orientation="h"  # 设置为水平方向
                      ))

    # Save & Show the plot
    fig.write_image(plot_results_path / 'cva.png')
    html_file_name = 'cva.html'
    py.plot(fig, filename=str(plot_results_path /
            html_file_name), auto_open=False)
    # html_content = pio.to_html(fig, include_plotlyjs='cdn')
    return html_file_name


def plot_dipca(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim, fault_labels):
    """DiPCA模型的结果可视化

    Args:
        phi_v_index (list): 动态综合指标φv,由T^2和SPE计算得到
        phi_s_index (list): 静态综合指标φs,由T^2和SPE计算得到
        phi_v_lim (float): 动态综合指标φv的阈值
        phi_s_lim (float): 静态综合指标φs的阈值
        fault_labels (list): 故障数据标签列

    Returns:
        html_file_name (str): 可视化结果的html文件名
    """

    N = len(phi_v_index)  # 获取数据长度
    fig = go.Figure()

    # 使用make_subplots创建包含三个子图的图表
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'φv', 'φs', 'Fault Labels'))

    # 添加第一个子图,显示φv
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=phi_v_index,
                             mode='lines', name='φv'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        phi_v_lim, phi_v_lim], mode='lines', name='φv threshold', line=dict(dash='dash')), row=1, col=1)

    # 添加第二个子图,显示φs
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=phi_s_index,
                             mode='lines', name='φs'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        phi_s_lim, phi_s_lim], mode='lines', name='φs threshold', line=dict(dash='dash')), row=2, col=1)

    # 添加第三个子图，显示故障数据标签列
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=fault_labels,
                             mode='markers+lines', name='Fault Labels',
                             marker={"size": 4, }), row=3, col=1)  # 点的大小

    # 更新布局
    fig.update_yaxes(type='log', title='φv', row=1, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=1, col=1, automargin=True)
    fig.update_yaxes(type='log', title='φs', row=2, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=2, col=1, automargin=True)
    fig.update_yaxes(title='Fault Labels', row=3, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=3, col=1, automargin=True)

    fig.update_layout(height=1000, width=1000,
                      legend=dict(
                          yanchor="bottom",
                          y=1.07,
                          xanchor="center",
                          x=0.5,
                          bgcolor='rgba(0,0,0,0)',  # 设置边框颜色为透明
                          borderwidth=0,  # 设置边框宽度为0，即隐藏边框
                          orientation="h"  # 设置为水平方向
                      ))
    # Save & Show the plot
    fig.write_image(plot_results_path / 'dipca.png')
    html_file_name = 'dipca.html'
    py.plot(fig, filename=str(plot_results_path /
            html_file_name), auto_open=False)
    # html_content = pio.to_html(fig, include_plotlyjs='cdn')
    return html_file_name


def plot_sfa(test_T2, test_T2e, test_S2, test_S2e,
             S2_threshold, S2e_threshold, T2_threshold, T2e_threshold, fault_labels):
    """SFA模型的结果可视化
    Args:
        test_T2 (list): T2统计量
        test_T2e (list): T2e统计量
        test_S2 (list): S2统计量
        test_S2e (list): S2e统计量
        S2_threshold (float): S2统计量阈值
        S2e_threshold (float): S2e统计量阈值
        T2_threshold (float): T2统计量阈值
        T2e_threshold (float): T2e统计量阈值

    Returns:
        html_file_name (str): 可视化结果的html文件名
    """

    N = len(test_T2)  # 获取数据长度

    fig = go.Figure()

    # 使用make_subplots创建包含五个子图的图表
    fig = make_subplots(rows=5, cols=1, subplot_titles=(
        'T^2', 'S^2', 'T^2e', "S^2e", 'Fault Labels'))

    # 添加第一个子图,显示T2
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=test_T2,
                             mode='lines', name='T2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        T2_threshold, T2_threshold], mode='lines', name='T2 threshold', line=dict(dash='dash')), row=1, col=1)

    # 添加第二个子图,显示S2
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=test_S2,
                             mode='lines', name='S2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        S2_threshold, S2_threshold], mode='lines', name='S2 threshold', line=dict(dash='dash')), row=2, col=1)

    # 添加第三个子图,显示T2e
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=test_T2e,
                             mode='lines', name='T2e'), row=3, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        T2e_threshold, T2e_threshold], mode='lines', name='T2e threshold', line=dict(dash='dash')), row=3, col=1)

    # 添加第四个子图,显示S2e
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=test_S2e,
                             mode='lines', name='S2e'), row=4, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        S2e_threshold, S2e_threshold], mode='lines', name='S2e threshold', line=dict(dash='dash')), row=4, col=1)

    # 添加第五个子图，显示故障数据标签列
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=fault_labels,
                             mode='markers+lines', name='Fault Labels',
                             marker={"size": 4, }), row=5, col=1)  # 点的大小

    # 更新布局
    fig.update_yaxes(type='log', title='T^2', row=1, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=1, col=1, automargin=True)
    fig.update_yaxes(type='log', title='S^2', row=2, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=2, col=1, automargin=True)
    fig.update_yaxes(type='log', title='T^2e', row=3, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=3, col=1, automargin=True)
    fig.update_yaxes(type='log', title='S^2e', row=4, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=4, col=1, automargin=True)
    fig.update_yaxes(title='Fault Labels', row=5, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=5, col=1, automargin=True)

    fig.update_layout(height=1000, width=1000,
                      legend=dict(
                          yanchor="bottom",
                          y=1.07,
                          xanchor="center",
                          x=0.5,
                          bgcolor='rgba(0,0,0,0)',  # 设置边框颜色为透明
                          borderwidth=0,  # 设置边框宽度为0，即隐藏边框
                          orientation="h"  # 设置为水平方向
                      ))

    # Save & Show the plot
    fig.write_image(plot_results_path / 'sfa.png')
    html_file_name = 'sfa.html'
    py.plot(fig, filename=str(plot_results_path /
            html_file_name), auto_open=False)
    # html_content = pio.to_html(fig, include_plotlyjs='cdn')
    return html_file_name


def plot_svdd(D_index, R_index, fault_labels):
    """DiPCA模型的结果可视化

    Args:
        D_index (list): 预测距离指标D
        R_index (float): 决策边界R(超球体半径)

    Returns:
        html_file_name (str): 可视化结果的html文件名
    """

    N = len(D_index)  # 获取数据长度
    fig = go.Figure()

    # 使用make_subplots创建包含两个子图的图表
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        'D index', 'Fault Labels'))

    # 添加第一个子图,显示φv
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=D_index,
                             mode='lines', name='D index'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, N], y=[
        R_index, R_index], mode='lines', name='R threshold', line=dict(dash='dash')), row=1, col=1)

    # 添加第二个子图，显示故障数据标签列
    fig.add_trace(go.Scatter(x=np.arange(1, N+1), y=fault_labels,
                             mode='markers+lines', name='Fault Labels',
                             marker={"size": 4, }), row=2, col=1)  # 点的大小

    # 更新布局
    fig.update_yaxes(type='log', title='D index',
                     row=1, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=1, col=1, automargin=True)
    fig.update_yaxes(title='Fault Labels', row=2, col=1, automargin=True)
    fig.update_xaxes(title='Sample Number', row=2, col=1, automargin=True)

    fig.update_layout(height=1000, width=1000,
                      legend=dict(
                          yanchor="bottom",
                          y=1.07,
                          xanchor="center",
                          x=0.5,
                          bgcolor='rgba(0,0,0,0)',  # 设置边框颜色为透明
                          borderwidth=0,  # 设置边框宽度为0，即隐藏边框
                          orientation="h"  # 设置为水平方向
                      ))
    # Save & Show the plot
    fig.write_image(plot_results_path / 'svdd.png')
    html_file_name = 'svdd.html'
    py.plot(fig, filename=str(plot_results_path /
            html_file_name), auto_open=False)
    # html_content = pio.to_html(fig, include_plotlyjs='cdn')
    return html_file_name


def cal_metrics(statistic, statistic_threshold, labels):
    """根据统计量结果,计算性能指标

    Args:
        statistic (numpy.ndarray): 统计量,如T^2,Q统计量等
        statistic_threshold (float): 统计量的阈值
        labels (numpy.ndarray): 故障数据标签列

    Returns:
        detectionRate (float): 检测率
        falseAlarmRate (float): 虚警率
        detectionTime (int): 首次检测成功时间-实际首次故障时间 / "ND"表示未检测到 / "OT"表示在故障发生前虚警
        [为避免误报,检测到的故障点需要连续显著高于阈值的时间实例的数量超过阈值]
    """
    detected = statistic > statistic_threshold  # 检测到的故障点

    if len(detected.shape) == 2:  # 如果detected为二维,则将其转为一维
        detected = np.squeeze(detected)

    labels = labels.astype(bool)  # 将 labels 转为 bool 类型 ,才能进行逻辑运算

    truePositives = np.sum(detected & labels)  # 正确检测的异常数
    actualFaults = np.sum(labels)  # 实际存在的异常数
    falsePositives = np.sum(detected & ~labels)  # 错误报警的次数
    normalInstances = np.sum(~labels)  # 正常情况的次数

    detectionRate = truePositives / actualFaults  # 检出率
    falseAlarmRate = falsePositives / normalInstances  # 虚警率

    firstTrueLabelIndex = np.where(labels == 1)[0][0]  # 第一个真实故障发生的索引

    thresholdConsecutiveSignificant = 80  # 连续显著高于阈值的时间实例的数量
    significantThreshold = statistic_threshold * 1.2  # 显著异常的阈值
    statistic = np.squeeze(statistic)  # 将 statistic 从二维变成一维

    detectedIndices = np.where(detected)[0]  # 检测到的故障点的索引
    if len(detectedIndices) == 0:
        detectionTime = "ND"
    else:
        flag_detect = 0
        consecutiveSignificant = 0
        for i in range(len(detectedIndices)):
            while detected[detectedIndices[i] + consecutiveSignificant]:  # and  : \
                # statistic[detectedIndices[i] + consecutiveSignificant] > significantThreshold:
                consecutiveSignificant = consecutiveSignificant + 1
                if detectedIndices[i] + consecutiveSignificant >= len(detected):
                    break
                if consecutiveSignificant >= thresholdConsecutiveSignificant:
                    firstDetectionIndex = detectedIndices[i]
                    detectionTime = firstDetectionIndex - firstTrueLabelIndex
                    # 从numpy.int64转为int,不然前端无法解析
                    detectionTime = int(detectionTime)
                    flag_detect = 1
                    break
            if flag_detect:
                break
        if not flag_detect:
            detectionTime = "ND"
    if detectionTime != 'ND':
        if detectionTime < 0:
            detectionTime = "OT"
    # 保留两位小数
    detectionRate = round(detectionRate, 2)
    falseAlarmRate = round(falseAlarmRate, 2)

    print("Metrics:", detectionRate, falseAlarmRate, detectionTime)

    return detectionRate, falseAlarmRate, detectionTime


@app.post("/run")
async def run(request_data: RunRequest):
    """运行模型

    Args:
        request_data (RunRequest): 
        请求数据,包含normal_name,fault_name,isReturnMetric,method,
        分别表示正常数据文件名,故障数据文件名,是否返回性能指标,模型名称

    Returns:
        json: 返回模型的结果
    """
    normal_name = request_data.normal_name
    fault_name = request_data.fault_name
    isReturnMetric = request_data.isReturnMetric
    method = request_data.method
    print(normal_name)
    print(fault_name)
    print("isReturnMetric:", isReturnMetric)
    print(method)

    normal_file_path = uploaded_files_path / "normal" / normal_name
    fault_file_path = uploaded_files_path / "fault" / fault_name

    # 使用 numpy 读取训练和测试数据文件
    normal_file = np.loadtxt(normal_file_path, delimiter=',')
    fault_file = np.loadtxt(fault_file_path, delimiter=',')
    # 提取前23列数据
    normal_data = normal_file[:, 0:23]
    fault_data = fault_file[:, 0:23]
    fault_labels = fault_file[:, -1]  # Label: 0表示正常，1表示故障

    # 根据Method值,调用不同的模型函数
    if method == "cva":

        # 数据转为matlab类型
        normal_data = matlab.double(normal_data.tolist())
        fault_data = matlab.double(fault_data.tolist())

        # 调用 CVA 模型
        alpha = 0.99  # 置信度
        n = 25        # 保留的状态维度
        p = 15        # 过去观测的长度
        f = 15        # 未来观测的长度
        [T2mon, Qmon, Ta, Qa] = eng.CVA(
            alpha, n, p, f, normal_data, fault_data, nargout=4)

        # 结果处理
        T2mon = np.array(T2mon).tolist()  # 从matlab类型转为python的list
        Qmon = np.array(Qmon).tolist()  # np类型shape是(1, 9076)形式的
        T2mon = T2mon[0]  # 巨坑！！转为list以后是还是二维的，所以需要先去掉一维
        Qmon = Qmon[0]

        # 结果可视化
        html_file_name = plot_cva(T2mon, Qmon, Ta, Qa, fault_labels)

        # 启动一个简单的HTTP服务器，为生成的HTML文件提供服务
        server_port = np.random.randint(10000, 65535)
        subprocess.Popen(["python", "-m", "http.server",
                          str(server_port)], cwd=plot_results_path)

        # 后端在本地服务器上生成HTML页面的URL,前端利用iframe标签引用 (方法5)
        server_url = f"http://localhost:{server_port}/{html_file_name}"
        print(server_url)

        # 性能指标计算
        detectionRateT2 = '/'
        falseAlarmRateT2 = '/'
        detectionTimeT2 = '/'
        detectionRateQ = '/'
        falseAlarmRateQ = '/'
        detectionTimeQ = '/'
        if isReturnMetric:
            # 计算性能指标
            # 首先调整label长度: 因为CVA用前p个样本预测第p+1个样本,所以Tmon和Qmon中的第1个样本对应原始样本中的第p个
            detectionRateT2, falseAlarmRateT2, detectionTimeT2 = cal_metrics(
                np.array(T2mon), Ta, fault_labels[p-1:])
            detectionRateQ, falseAlarmRateQ, detectionTimeQ = cal_metrics(
                np.array(Qmon), Qa, fault_labels[p-1:])

        return {
            "html_url": server_url,
            "metric": {
                "detectionRateT2": detectionRateT2,
                "falseAlarmRateT2": falseAlarmRateT2,
                "detectionTimeT2": detectionTimeT2,
                "detectionRateQ": detectionRateQ,
                "falseAlarmRateQ": falseAlarmRateQ,
                "detectionTimeQ": detectionTimeQ
            }
        }

        # (方法4) 返回生成plotly图表的静态图片,前端直接展示图片(无法交互了)
        # (方法3) 返回生成plotly图表的json结果,前端再利用plotly.js进行可视化
        # (方法2) 直接返回html内容
        # (方法1) 返回模型的结果，前端进行可视化

    elif method == "dipca":

        # 调用 DiPCA 模型
        s = 2  # s为滞后阶数
        a = 5  # a为主成分数量
        phi_v_index, phi_s_index, phi_v_lim, phi_s_lim = runDiPCA(
            normal_data, fault_data, s, a)

        # 结果可视化
        html_file_name = plot_dipca(
            phi_v_index, phi_s_index, phi_v_lim, phi_s_lim, fault_labels)
        # 启动一个简单的HTTP服务器，为生成的HTML文件提供服务
        server_port = np.random.randint(10000, 65535)
        subprocess.Popen(["python", "-m", "http.server",
                          str(server_port)], cwd=plot_results_path)

        # 后端在本地服务器上生成HTML页面的URL,前端利用iframe标签引用
        server_url = f"http://localhost:{server_port}/{html_file_name}"
        print(server_url)

        # 性能指标计算
        detectionRatePhiV = '/'
        falseAlarmRatePhiV = '/'
        detectionTimePhiV = '/'
        detectionRatePhiS = '/'
        falseAlarmRatePhiS = '/'
        detectionTimePhiS = '/'
        # 计算性能指标
        if isReturnMetric:
            # 首先调整label长度: 因为DiPCA的滞后系数为s,所以Tmon和Qmon中的第1个样本对应原始样本中的第s+1个
            detectionRatePhiV, falseAlarmRatePhiV, detectionTimePhiV = cal_metrics(
                phi_v_index, phi_v_lim, fault_labels[s:])
            detectionRatePhiS, falseAlarmRatePhiS, detectionTimePhiS = cal_metrics(
                phi_s_index, phi_s_lim, fault_labels[s:])

        return {
            "html_url": server_url,
            "metric": {
                "detectionRatePhiV": detectionRatePhiV,
                "falseAlarmRatePhiV": falseAlarmRatePhiV,
                "detectionTimePhiV": detectionTimePhiV,
                "detectionRatePhiS": detectionRatePhiS,
                "falseAlarmRatePhiS": falseAlarmRatePhiS,
                "detectionTimePhiS": detectionTimePhiS
            }
        }

    elif method == 'sfa':

        # 调用 SFA 模型
        test_T2, test_T2e, test_S2, test_S2e,\
            S2_threshold, S2e_threshold, T2_threshold, T2e_threshold = runSFA(
                normal_data, fault_data)

        # 结果可视化
        html_file_name = plot_sfa(
            test_T2, test_T2e, test_S2, test_S2e,
            S2_threshold, S2e_threshold, T2_threshold, T2e_threshold, fault_labels
        )
        # 启动一个简单的HTTP服务器，为生成的HTML文件提供服务
        server_port = np.random.randint(10000, 65535)
        subprocess.Popen(["python", "-m", "http.server",
                          str(server_port)], cwd=plot_results_path)

        # 后端在本地服务器上生成HTML页面的URL,前端利用iframe标签引用
        server_url = f"http://localhost:{server_port}/{html_file_name}"
        print(server_url)

        # 性能指标计算
        detectionRateT2 = '/'
        falseAlarmRateT2 = '/'
        detectionTimeT2 = '/'
        detectionRateT2e = '/'
        falseAlarmRateT2e = '/'
        detectionTimeT2e = '/'
        detectionRateS2 = '/'
        falseAlarmRateS2 = '/'
        detectionTimeS2 = '/'
        detectionRateS2e = '/'
        falseAlarmRateS2e = '/'
        detectionTimeS2e = '/'
        # 计算性能指标
        if isReturnMetric:
            detectionRateT2, falseAlarmRateT2, detectionTimeT2 = cal_metrics(
                test_T2, T2_threshold, fault_labels)
            detectionRateT2e, falseAlarmRateT2e, detectionTimeT2e = cal_metrics(
                test_T2e, T2e_threshold, fault_labels)
            detectionRateS2, falseAlarmRateS2, detectionTimeS2 = cal_metrics(
                test_S2, S2_threshold, fault_labels)
            detectionRateS2e, falseAlarmRateS2e, detectionTimeS2e = cal_metrics(
                test_S2e, S2e_threshold, fault_labels)

        return {
            "html_url": server_url,
            "metric": {
                "detectionRateT2": detectionRateT2,
                "falseAlarmRateT2": falseAlarmRateT2,
                "detectionTimeT2": detectionTimeT2,
                "detectionRateT2e": detectionRateT2e,
                "falseAlarmRateT2e": falseAlarmRateT2e,
                "detectionTimeT2e": detectionTimeT2e,
                "detectionRateS2": detectionRateS2,
                "falseAlarmRateS2": falseAlarmRateS2,
                "detectionTimeS2": detectionTimeS2,
                "detectionRateS2e": detectionRateS2e,
                "falseAlarmRateS2e": falseAlarmRateS2e,
                "detectionTimeS2e": detectionTimeS2e
            }
        }

    elif method == 'dipls':
        # 数据预处理
        # 将第8,9,21,22作为Y变量,其余作为X变量
        X_Train = normal_data[:, [i for i in range(
            normal_data.shape[1]) if i not in [7, 8, 20, 21]]]
        Y_Train = normal_data[:, [7, 8, 20, 21]]
        X_test = fault_data[:, [i for i in range(
            fault_data.shape[1]) if i not in [7, 8, 20, 21]]]
        Y_test = fault_data[:, [7, 8, 20, 21]]

        # 调用 DiPLS 模型
        s = 4  # 滞后阶数
        a = 2  # 潜变量个数
        T2mon, Qmon, Ta, Qa = runDiPLS(X_Train, Y_Train, X_test, Y_test, s, a)

        # 结果可视化
        # 由于DiPLS的结果与CVA相同,所以直接调用CVA的可视化函数
        html_file_name = plot_cva(T2mon, Qmon, Ta, Qa, fault_labels)

        # 启动一个简单的HTTP服务器，为生成的HTML文件提供服务
        server_port = np.random.randint(10000, 65535)
        subprocess.Popen(["python", "-m", "http.server",
                          str(server_port)], cwd=plot_results_path)

        # 后端在本地服务器上生成HTML页面的URL,前端利用iframe标签引用 (方法5)
        server_url = f"http://localhost:{server_port}/{html_file_name}"
        print(server_url)

        # 性能指标计算
        detectionRateT2 = '/'
        falseAlarmRateT2 = '/'
        detectionTimeT2 = '/'
        detectionRateQ = '/'
        falseAlarmRateQ = '/'
        detectionTimeQ = '/'
        if isReturnMetric:
            # 计算性能指标
            # 首先调整label长度: 因为DiPLS的滞后系数为s,所以Tmon和Qmon中的第1个样本对应原始样本中的第s+1个
            detectionRateT2, falseAlarmRateT2, detectionTimeT2 = cal_metrics(
                np.array(T2mon), Ta, fault_labels)
            detectionRateQ, falseAlarmRateQ, detectionTimeQ = cal_metrics(
                np.array(Qmon), Qa, fault_labels)

        return {
            "html_url": server_url,
            "metric": {
                "detectionRateT2": detectionRateT2,
                "falseAlarmRateT2": falseAlarmRateT2,
                "detectionTimeT2": detectionTimeT2,
                "detectionRateQ": detectionRateQ,
                "falseAlarmRateQ": falseAlarmRateQ,
                "detectionTimeQ": detectionTimeQ
            }
        }
    elif method == 'svdd':
     # 数据转为matlab类型
        normal_data = matlab.double(normal_data.tolist())
        fault_data = matlab.double(fault_data.tolist())

        # 调用 SVDD 模型
        [D_index, R_index] = eng.SVDD(normal_data, fault_data, nargout=2)

        # 结果处理
        D_index = np.array(D_index).tolist()  # 从matlab类型转为python的list
        D_index = D_index[0]

        # 结果可视化
        html_file_name = plot_svdd(D_index, R_index, fault_labels)

        # 启动一个简单的HTTP服务器，为生成的HTML文件提供服务
        server_port = np.random.randint(10000, 65535)
        subprocess.Popen(["python", "-m", "http.server",
                          str(server_port)], cwd=plot_results_path)

        # 后端在本地服务器上生成HTML页面的URL,前端利用iframe标签引用 (方法5)
        server_url = f"http://localhost:{server_port}/{html_file_name}"
        print(server_url)

        # 性能指标计算
        detectionRateD = '/'
        falseAlarmRateD = '/'
        detectionTimeD = '/'
        if isReturnMetric:
            # 计算性能指标
            # 首先调整label长度:
            detectionRateD, falseAlarmRateD, detectionTimeD = cal_metrics(
                np.array(D_index), R_index, fault_labels)

        return {
            "html_url": server_url,
            "metric": {
                "detectionRateD": detectionRateD,
                "falseAlarmRateD": falseAlarmRateD,
                "detectionTimeD": detectionTimeD,
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
