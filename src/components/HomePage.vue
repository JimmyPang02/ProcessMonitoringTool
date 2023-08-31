<!-- eslint-disable vue/require-v-for-key -->
<template>
  <a-spin v-if="isProcessing" size="large"
    style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);"></a-spin>
  <h2>Process Monitoring Tool</h2>
  <div>
    <div class="workspace-container">
      <!-- 正常数据 -->
      <div class="data-card">
        <div>
          <h3>正常数据</h3>
        </div>
        <!-- 上传按钮 -->
        <div style="margin-top: 10px;margin-bottom: 10px;">
          <a-upload-dragger :multiple="true" action="#" :before-upload="handleNormalFileUpload" accept=".csv"
            :show-upload-list="false">
            <p class="ant-upload-drag-icon">
              <upload-outlined></upload-outlined>
            </p>
            <p class="ant-upload-text" style="font-size: small;">单击或拖动文件到此区域进行上传</p>
            <p class="ant-upload-hint" style="font-size: small;">
              支持单个或批量上传
            </p>
          </a-upload-dragger>
        </div>

        <!-- 训练数据显示 -->
        <a-list>
          <!-- 添加鼠标悬浮事件处理 -->
          <a-list-item v-for="(fileName, index) in normalFiles" :key="index" @click="selectNormalFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedNormalIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <!-- 打钩icon -->
            <span class="check-icon" v-if="selectedNormalIndex === index">
              <CheckSquareOutlined />
            </span>
            {{ fileName }}
            <!-- 垃圾桶icon -->
            <span class="trash-button" v-if="hoveredItem === index" type="delete"
              @click.stop="deleteFile(fileName, 'normal', index)">
              <DeleteOutlined />
            </span>
          </a-list-item>
        </a-list>
      </div>

      <!-- 故障数据 -->
      <div class=" data-card">
        <div>
          <h3>故障数据
            <a-tooltip>
              <template #title>
                1. 故障数据表应包含一个标签列,用来识别每个时间点的故障情况(如是否发生故障或者故障程度)。<br>
                2. 标签列应该放在故障数据表格的最后一列。
              </template>
              <QuestionCircleOutlined />
            </a-tooltip>
          </h3>
          <!-- 故障数据说明卡片 -->

        </div>
        <div style="margin-top: 10px;margin-bottom: 10px;">
          <a-upload-dragger :multiple="true" action="#" :before-upload="handleFaultFileUpload" accept=".csv"
            :show-upload-list="false">
            <p class="ant-upload-drag-icon">
              <upload-outlined></upload-outlined>
            </p>
            <p class="ant-upload-text" style="font-size: small;">单击或拖动文件到此区域进行上传</p>
            <p class="ant-upload-hint" style="font-size: small;">
              支持单个或批量上传
            </p>
          </a-upload-dragger>
        </div>

        <!-- 故障数据显示 -->
        <a-list>
          <!-- 添加鼠标悬浮事件处理 -->
          <a-list-item v-for="(fileName, index) in faultFiles" :key="index" @click="selectFaultFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedFaultIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <div>
              <!-- 打钩icon -->
              <span class="check-icon" v-if="selectedFaultIndex === index">
                <CheckSquareOutlined />
              </span>
              <!-- 打钩icon -->
              {{ fileName }}
            </div>
            <!-- 垃圾桶icon -->
            <span class="trash-button" v-if="hoveredItem === index" type="delete"
              @click.stop="deleteFile(fileName, 'fault', index)">
              <DeleteOutlined />
            </span>

          </a-list-item>
        </a-list>
      </div>

      <!-- 算法选择和运行 -->
      <div class="process-card">
        <div class="process-controls">
          选择算法：
          <a-select show-search v-model:value="selectedOption" class="option-select" placeholder="CVA">
            <a-select-option value="cva">CVA</a-select-option>
            <a-select-option value="dipca">DiPCA</a-select-option>
            <a-select-option value="dipls">DiPLS</a-select-option>
            <a-select-option value="sfa">SFA</a-select-option>
            <a-select-option value="svdd">SVDD</a-select-option>
            <a-select-option value="ae">AE</a-select-option>
          </a-select>
          <a-button type="primary" @click="runProcess" class="run-button">运行</a-button>
          <a-button @click="showModal" style="margin-left: 10px;">设置</a-button>
          <a-modal v-model:visible="visible" title="设置" cancelText="取消" okText="确定" @ok="handleOk" @cancel="handleCancel">
            <a-card>
              <h3><b>返回算法性能指标</b> <a-checkbox v-model:checked="isReturnMetric" style="float: right;"></a-checkbox></h3>
              注意: 选择此项必须保证故障数据的标签数据是01变量,即只有故障和正常两种状态,才能计算检出率和虚警率等指标
            </a-card>
          </a-modal>
        </div>

        <div class="chart-output">
          <!-- 图表输出区域 -->
          <iframe :src="chartUrl" frameborder="0" width="100%" height="100%" style="overflow: hidden;"></iframe>
        </div>
        <div class="table-output">
          <!-- 文本输出区域 -->
          <a-table :dataSource="dataSource" :columns="columns" :pagination="false" width="100%" height="100%" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Button, UploadDragger, Select, Spin } from 'ant-design-vue';
import { List } from 'ant-design-vue';
import { ref, onMounted } from 'vue';
import { UploadOutlined, CheckSquareOutlined, DeleteOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue';
import API_ROUTES from "@/api/api";
import { backendIP } from "@/api/backend";
import axios from 'axios';
import { Modal } from 'ant-design-vue';


export default {
  name: 'HomePage',
  components: {
    ASpin: Spin,
    ASelect: Select,
    AUploadDragger: UploadDragger,
    AButton: Button,
    AList: List,
    AListItem: List.Item,
    UploadOutlined, CheckSquareOutlined, DeleteOutlined, QuestionCircleOutlined
  },
  setup() {
    // 设置 axios 的 baseURL
    axios.defaults.baseURL = backendIP;

    // 是否正在运行过程监测
    const isProcessing = ref(false);

    // 算法选择
    const selectedOption = ref('cva');

    // 文件上传相关
    const normalFiles = ref([]);
    const faultFiles = ref([]);

    // 是否悬浮【a-list-item 并没有内置用于判断鼠标是否悬浮的 API，所以利用mouseenter等实现】
    const hoveredItem = ref(null);

    // 点击选中的训练集和测试集索引
    const selectedNormalIndex = ref(null);
    const selectedFaultIndex = ref(null);

    // 图表输出区域的
    let chartUrl = ref('');

    // 表格输出区域的数据
    const dataSource = ref([
      {
        'method': '/',
        'detectionRateT2': '/',
        'falseAlarmRateT2': '/',
        'detectionTimeT2': '/',
        'detectionRateQ': '/',
        'falseAlarmRateQ': '/',
        'detectionTimeQ': '/',
      },
    ]);
    const columns = ref([
      {
        title: 'method', dataIndex: 'method', key: 'method',
      },
      {
        title: 'T^2 detection rate (%)', dataIndex: 'detectionRateT2', key: 'detectionRateT2',
      },
      {
        title: 'T^2 false alarm rate (%)', dataIndex: 'falseAlarmRateT2', key: 'falseAlarmRateT2',
      },
      {
        title: 'T^2 detection time (s)', dataIndex: 'detectionTimeT2', key: 'detectionTimeT2',
      },
      {
        title: 'Q detection rate (%)', dataIndex: 'detectionRateQ', key: 'detectionRateQ',
      },
      {
        title: 'Q false alarm rate (%)', dataIndex: 'falseAlarmRateQ', key: 'falseAlarmRateQ',
      },
      {
        title: 'Q detection time (s)', dataIndex: 'detectionTimeQ', key: 'detectionTimeQ',
      }
    ]);

    // 处理模态框的显示和隐藏
    const visible = ref(false);
    const showModal = () => {
      visible.value = true;
    };
    const handleOk = () => {
      visible.value = false;
    };
    const handleCancel = () => {
      visible.value = false;
    };

    // 是否返回检出率等指标
    const isReturnMetric = ref(false);

    // 处理列表项的鼠标悬浮事件
    const handleListItemHover = (isHovered, index) => {
      hoveredItem.value = isHovered ? index : null;
    }

    // 训练文件上传
    const handleNormalFileUpload = async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post(API_ROUTES.uploadData, formData, {
          params: { type: "normal" }
        });
        if (response.data.message === 'success') {
          normalFiles.value.push(file.name);
        }
      } catch (error) {
        console.error('上传训练文件时出现错误：', error);
      }
      return false;
    };

    // 测试文件上传
    const handleFaultFileUpload = async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post(API_ROUTES.uploadData, formData, {
          params: { type: "fault" }
        });
        if (response.data.message === 'success') {
          faultFiles.value.push(file.name);
        }
      } catch (error) {
        console.error('上传测试文件时出现错误：', error);
      }
      return false;
    };


    // 获取数据列表
    const fetchFileList = async (data_type) => {
      try {
        const response = await axios.get(API_ROUTES.DataList, {
          params: { type: data_type }
        });
        if (response.data.message === 'success') {
          return response.data.data;
        }
      } catch (error) {
        console.error('获取数据列表时出现错误：', error);
      }
      return [];
    };

    // 在组件加载后获取数据列表
    onMounted(async () => {
      normalFiles.value = await fetchFileList('normal');
      faultFiles.value = await fetchFileList('fault');
    });

    // 删除数据
    const deleteFile = async (fileName, data_type, index) => {
      try {
        await axios.get(API_ROUTES.deleteData, {
          params: { file_name: fileName, type: data_type }
        });
        if (data_type === 'normal') {
          normalFiles.value.splice(index, 1);
        } else if (data_type === 'fault') {
          faultFiles.value.splice(index, 1);
        }
      } catch (error) {
        console.error('删除文件时出现错误：', error);
      }
    };

    // 选中训练数据
    const selectNormalFile = (index) => {
      if (selectedNormalIndex.value === index) {
        // 已经选中，取消选中
        selectedNormalIndex.value = null;
      } else {
        // 未选中，选中它
        selectedNormalIndex.value = index;
      }
    };

    // 选中测试数据
    const selectFaultFile = (index) => {
      if (selectedFaultIndex.value === index) {
        // 已经选中，取消选中
        selectedFaultIndex.value = null;
      } else {
        // 未选中，选中它
        selectedFaultIndex.value = index;
      }
    };


    // 过程监测API请求函数
    const runPM = async (normal_name, fault_name, method) => {
      try {
        const response = await axios.post(API_ROUTES.Run, {
          normal_name: normal_name,
          fault_name: fault_name,
          isReturnMetric: isReturnMetric.value,
          method: method
        },
        );
        console.log(response.data);
        return response.data;
      } catch (error) {
        console.error('运行PM出错:', error);
        isProcessing.value = false;
      }
    };


    // 运行过程监测的逻辑
    const runProcess = async () => {

      isProcessing.value = true; // 开始加载动画

      if (selectedNormalIndex.value === null || selectedFaultIndex.value === null) {
        // 如果没有选中训练集或测试集，弹窗提醒用户选择数据集
        Modal.warning({
          title: '请先选择正常数据和故障数据',
          content: '点击数据集的名称,选择正常数据和故障数据,用于过程监测模型的训练和测试。',
        });
        return; // 不继续执行下面的操作
      }
      console.log('runProcess')

      // 在运行过程中，使用选中的训练集索引和测试集索引来获取对应的文件名
      const selectedNormalFileName = normalFiles.value[selectedNormalIndex.value];
      const selectedFaultFileName = faultFiles.value[selectedFaultIndex.value];

      // 执行模型训练和推理操作
      if (selectedOption.value == 'cva') {
        console.log('cva')

        // 运行CVA
        const res = await runPM(selectedNormalFileName, selectedFaultFileName, 'cva');

        // 接收到后端返回的图表链接
        chartUrl.value = res.html_url;

        // 定义表格列
        columns.value = [
          {
            title: 'method', dataIndex: 'method', key: 'method',
          },
          {
            title: 'T^2 detection rate (%)', dataIndex: 'detectionRateT2', key: 'detectionRateT2',
          },
          {
            title: 'T^2 false alarm rate (%)', dataIndex: 'falseAlarmRateT2', key: 'falseAlarmRateT2',
          },
          {
            title: 'T^2 detection time (s)', dataIndex: 'detectionTimeT2', key: 'detectionTimeT2',
          },
          {
            title: 'Q detection rate (%)', dataIndex: 'detectionRateQ', key: 'detectionRateQ',
          },
          {
            title: 'Q false alarm rate (%)', dataIndex: 'falseAlarmRateQ', key: 'falseAlarmRateQ',
          },
          {
            title: 'Q detection time (s)', dataIndex: 'detectionTimeQ', key: 'detectionTimeQ',
          }
        ];
        // 接收到后端返回的表格数据
        dataSource.value = [
          {
            'method': selectedOption.value,
            'detectionRateT2': res.metric.detectionRateT2,
            'falseAlarmRateT2': res.metric.falseAlarmRateT2,
            'detectionTimeT2': res.metric.detectionTimeT2,
            'detectionRateQ': res.metric.detectionRateQ,
            'falseAlarmRateQ': res.metric.falseAlarmRateQ,
            'detectionTimeQ': res.metric.detectionTimeQ,
          },
        ];
      }
      else if (selectedOption.value == 'dipca') {
        console.log('DiPCA')
        // 运行DiPCA
        const res = await runPM(selectedNormalFileName, selectedFaultFileName, 'dipca');

        // 接收到后端返回的图表链接
        chartUrl.value = res.html_url;

        // 定义表格列
        columns.value = [
          {
            title: 'method', dataIndex: 'method', key: 'method',
          },
          {
            title: 'Phi_v detection rate (%)', dataIndex: 'detectionRatePhi_v', key: 'detectionRatePhi_v',
          },
          {
            title: 'Phi_v false alarm rate (%)', dataIndex: 'falseAlarmRatePhi_v', key: 'falseAlarmRatePhi_v',
          },
          {
            title: 'Phi_v detection time (s)', dataIndex: 'detectionTimePhi_v', key: 'detectionTimePhi_v',
          },
          {
            title: 'Phi_s detection rate (%)', dataIndex: 'detectionRatePhi_s', key: 'detectionRatePhi_s',
          },
          {
            title: 'Phi_s false alarm rate (%)', dataIndex: 'falseAlarmRatePhi_s', key: 'falseAlarmRatePhi_s',
          },
          {
            title: 'Phi_s detection time (s)', dataIndex: 'detectionTimePhi_s', key: 'detectionTimePhi_s',
          }
        ];
        // 接收到后端返回的表格数据
        dataSource.value = [
          {
            'method': 'DiPCA',
            'detectionRatePhi_v': res.metric.detectionRatePhiV,
            'falseAlarmRatePhi_v': res.metric.falseAlarmRatePhiV,
            'detectionTimePhi_v': res.metric.detectionTimePhiV,
            'detectionRatePhi_s': res.metric.detectionRatePhiS,
            'falseAlarmRatePhi_s': res.metric.falseAlarmRatePhiS,
            'detectionTimePhi_s': res.metric.detectionTimePhiS,
          },
        ];
      }
      else if (selectedOption.value == 'dipls') {
        console.log('DiPLS')
        // 运行DiPCA
        const res = await runPM(selectedNormalFileName, selectedFaultFileName, 'dipls');

        // 接收到后端返回的图表链接
        chartUrl.value = res.html_url;

        // 定义表格列
        columns.value = [
          {
            title: 'method', dataIndex: 'method', key: 'method',
          },
          {
            title: 'T^2 detection rate (%)', dataIndex: 'detectionRateT2', key: 'detectionRateT2',
          },
          {
            title: 'T^2 false alarm rate (%)', dataIndex: 'falseAlarmRateT2', key: 'falseAlarmRateT2',
          },
          {
            title: 'T^2 detection time (s)', dataIndex: 'detectionTimeT2', key: 'detectionTimeT2',
          },
          {
            title: 'Q detection rate (%)', dataIndex: 'detectionRateQ', key: 'detectionRateQ',
          },
          {
            title: 'Q false alarm rate (%)', dataIndex: 'falseAlarmRateQ', key: 'falseAlarmRateQ',
          },
          {
            title: 'Q detection time (s)', dataIndex: 'detectionTimeQ', key: 'detectionTimeQ',
          }
        ];
        // 接收到后端返回的表格数据
        dataSource.value = [
          {
            'method': 'DiPLS',
            'detectionRateT2': res.metric.detectionRateT2,
            'falseAlarmRateT2': res.metric.falseAlarmRateT2,
            'detectionTimeT2': res.metric.detectionTimeT2,
            'detectionRateQ': res.metric.detectionRateQ,
            'falseAlarmRateQ': res.metric.falseAlarmRateQ,
            'detectionTimeQ': res.metric.detectionTimeQ,
          },
        ];
      }

      else if (selectedOption.value == 'sfa') {
        console.log('SFA')
        // 运行SFA
        const res = await runPM(selectedNormalFileName, selectedFaultFileName, 'sfa');

        // 接收到后端返回的图表链接
        chartUrl.value = res.html_url;

        // 定义表格列
        columns.value = [
          {
            title: 'method', dataIndex: 'method', key: 'method',
          },
          {
            title: 'T^2 detection rate (%)', dataIndex: 'detectionRateT2', key: 'detectionRateT2',
          },
          {
            title: 'T^2 false alarm rate (%)', dataIndex: 'falseAlarmRateT2', key: 'falseAlarmRateT2',
          },
          {
            title: 'T^2 detection time (s)', dataIndex: 'detectionTimeT2', key: 'detectionTimeT2',
          },
          {
            title: 'S^2 detection rate (%)', dataIndex: 'detectionRateS2', key: 'detectionRateS2',
          },
          {
            title: 'S^2 false alarm rate (%)', dataIndex: 'falseAlarmRateS2', key: 'falseAlarmRateS2',
          },
          {
            title: 'S^2 detection time (s)', dataIndex: 'detectionTimeS2', key: 'detectionTimeS2',
          },
          // {
          //   title: 'T^2e detection rate (%)', dataIndex: 'detectionRateT2e', key: 'detectionRateT2e',
          // },
          // {
          //   title: 'T^2e false alarm rate (%)', dataIndex: 'falseAlarmRateT2e', key: 'falseAlarmRateT2e',
          // },
          // {
          //   title: 'T^2e detection time (s)', dataIndex: 'detectionTimeT2e', key: 'detectionTimeT2e',
          // },
          // {
          //   title: 'S^2e detection rate (%)', dataIndex: 'detectionRateS2e', key: 'detectionRateS2e',
          // },
          // {
          //   title: 'S^2e false alarm rate (%)', dataIndex: 'falseAlarmRateS2e', key: 'falseAlarmRateS2e',
          // },
          // {
          //   title: 'S^2e detection time (s)', dataIndex: 'detectionTimeS2e', key: 'detectionTimeS2e',
          // }
        ];
        // 接收到后端返回的表格数据
        dataSource.value = [
          {
            'method': 'SFA',
            'detectionRateT2': res.metric.detectionRateT2,
            'falseAlarmRateT2': res.metric.falseAlarmRateT2,
            'detectionTimeT2': res.metric.detectionTimeT2,
            'detectionRateS2': res.metric.detectionRateS2,
            'falseAlarmRateS2': res.metric.falseAlarmRateS2,
            'detectionTimeS2': res.metric.detectionTimeS2,
            // 'detectionRateT2e': res.metric.detectionRateT2e,
            // 'falseAlarmRateT2e': res.metric.falseAlarmRateT2e,
            // 'detectionTimeT2e': res.metric.detectionTimeT2e,
            // 'detectionRateS2e': res.metric.detectionRateS2e,
            // 'falseAlarmRateS2e': res.metric.falseAlarmRateS2e,
            // 'detectionTimeS2e': res.metric.detectionTimeS2e,
          },
        ]
      }
      else if (selectedOption.value == 'svdd') {
        console.log('SVDD')

        // 运行SVDD
        const res = await runPM(selectedNormalFileName, selectedFaultFileName, 'svdd');

        // 接收到后端返回的图表链接
        chartUrl.value = res.html_url;

        // 定义表格列
        columns.value = [
          {
            title: 'method', dataIndex: 'method', key: 'method',
          },
          {
            title: 'D detection rate (%)', dataIndex: 'detectionRateD', key: 'detectionRateD',
          },
          {
            title: 'D false alarm rate (%)', dataIndex: 'falseAlarmRateD', key: 'falseAlarmRateD',
          },
          {
            title: 'D detection time (s)', dataIndex: 'detectionTimeD', key: 'detectionTimeD',
          },
        ];
        // 接收到后端返回的表格数据
        dataSource.value = [
          {
            'method': 'SVDD',
            'detectionRateD': res.metric.detectionRateD,
            'falseAlarmRateD': res.metric.falseAlarmRateD,
            'detectionTimeD': res.metric.detectionTimeD,
          },
        ]
      }
      else if (selectedOption.value == 'ae') {
        console.log('AE')
      }
      else {
        console.log('method error')
      }

      isProcessing.value = false; // 结束加载动画
    };

    // 返回需要暴露给模板的数据和方法
    return {
      isProcessing,
      isReturnMetric,
      visible,
      showModal,
      handleOk,
      handleCancel,
      dataSource,
      columns,
      chartUrl,
      runPM,
      fetchFileList,
      handleListItemHover,
      hoveredItem,
      selectFaultFile,
      selectNormalFile,
      selectedFaultIndex,
      selectedNormalIndex,
      normalFiles,
      faultFiles,
      handleNormalFileUpload,
      handleFaultFileUpload,
      selectedOption,
      runProcess,
      deleteFile
    };
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.workspace-container {
  display: flex;
  justify-content: space-between;
  padding-left: 10px;
  padding-right: 10px;

}

.data-card {
  flex: 1;
  padding: 20px;
  border: 1px solid #ccc;
  background-color: #f7f7f7;
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  margin-right: 20px;
  flex-direction: column;
  /* To stack title and upload component vertically */
}

.process-card {
  flex: 8;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 20px;
  border: 1px solid #ccc;
  background-color: #f7f7f7;
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
}

.process-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  /*居中*/
  margin-bottom: 10px;
}

.option-select {
  width: 50%;
  margin-right: 20px;
}

.run-button {
  flex-shrink: 0;
}

.chart-output {
  height: 60vh;
  border: 1px solid #ccc;
  background-color: white;
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  padding: 5px;
}

.table-output {

  margin-top: 20px;
}



.a-upload-dragger {
  height: 60px;

}

.list-item-content:hover {
  box-shadow: 0 0 8px lightgray;
  position: relative;
}

.trash-button {
  position: absolute;
  right: 5px;
  top: 50%;
  transform: translateY(-50%);
  display: none;
  font-size: 16px;
  color: red;
}

.list-item-content:hover .trash-button {
  display: block;
}

.highlighted {
  background-color: #30649c;
  color: white;
}

.check-icon {
  font-size: 16px;
  color: #ffffff;
}
</style>
