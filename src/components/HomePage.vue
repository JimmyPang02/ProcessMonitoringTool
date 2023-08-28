<template>
  <h2>Process Monitoring Tool</h2>
  <div>
    <div class="workspace-container">
      <!-- 训练数据 -->
      <div class="data-card">
        <div>
          <h3>训练数据</h3>
        </div>
        <!-- 上传按钮 -->
        <div style="margin-top: 10px;margin-bottom: 10px;">
          <a-upload-dragger :multiple="true" action="#" :before-upload="handleTrainFileUpload" accept=".csv"
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
          <a-list-item v-for="(fileName, index) in trainingFiles" :key="index" @click="selectTrainingFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedTrainingIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <!-- 打钩icon -->
            <span class="check-icon" v-if="selectedTrainingIndex === index">
              <CheckSquareOutlined />
            </span>
            {{ fileName }}
            <!-- 垃圾桶icon -->
            <span class="trash-button" v-if="hoveredItem === index" type="delete"
              @click.stop="deleteFile(fileName, 'train', index)">
              <DeleteOutlined />
            </span>
          </a-list-item>
        </a-list>
      </div>

      <!-- 测试数据 -->
      <div class=" data-card">
        <div>
          <h3>测试数据</h3>
        </div>
        <div style="margin-top: 10px;margin-bottom: 10px;">
          <a-upload-dragger :multiple="true" action="#" :before-upload="handleTestFileUpload" accept=".csv"
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

        <!-- 测试数据显示 -->
        <a-list>
          <!-- 添加鼠标悬浮事件处理 -->
          <a-list-item v-for="(fileName, index) in testingFiles" :key="index" @click="selectTestingFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedTestingIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <div>
              <!-- 打钩icon -->
              <span class="check-icon" v-if="selectedTestingIndex === index">
                <CheckSquareOutlined />
              </span>
              <!-- 打钩icon -->
              {{ fileName }}
            </div>
            <!-- 垃圾桶icon -->
            <span class="trash-button" v-if="hoveredItem === index" type="delete"
              @click.stop="deleteFile(fileName, 'test', index)">
              <DeleteOutlined />
            </span>

          </a-list-item>
        </a-list>
      </div>

      <!-- 算法选择和运行 -->
      <div class="process-card">
        <div class="process-controls">
          选择算法：
          <a-select show-search v-model:value="selectedOption" class="option-select" placeholder="PCA">
            <a-select-option value="pca">PCA</a-select-option>
            <a-select-option value="cva">CVA</a-select-option>
            <a-select-option value="ae">AE</a-select-option>
          </a-select>
          <a-button type="primary" @click="runProcess" class="run-button">运行</a-button>
        </div>

        <div class="chart-output">
          <!-- 图表输出区域 -->
          <div v-html="plotHTML"></div>
          <iframe :src="chartUrl" frameborder="0" width="800" height="600"></iframe>
        </div>
        <div v-html="plotHTML.value"></div>

        <div class="text-output">
          <!-- 文本输出区域 -->
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Button, UploadDragger, Select } from 'ant-design-vue';
import { List } from 'ant-design-vue';
import { ref, onMounted } from 'vue';
import { UploadOutlined, CheckSquareOutlined, DeleteOutlined } from '@ant-design/icons-vue';
import API_ROUTES from "@/api/api";
import { backendIP } from "@/api/backend";
import axios from 'axios';

export default {
  name: 'HomePage',
  components: {
    ASelect: Select,
    AUploadDragger: UploadDragger,
    AButton: Button,
    AList: List,
    AListItem: List.Item,
    UploadOutlined, CheckSquareOutlined, DeleteOutlined
  },
  setup() {
    // 设置 axios 的 baseURL
    axios.defaults.baseURL = backendIP;

    // 算法选择
    const selectedOption = ref('pca');

    // 文件上传相关
    const trainingFiles = ref([]);
    const testingFiles = ref([]);

    // 是否悬浮【a-list-item 并没有内置用于判断鼠标是否悬浮的 API，所以利用mouseenter等实现】
    const hoveredItem = ref(null);

    // 点击选中的训练集和测试集索引
    const selectedTrainingIndex = ref(null);
    const selectedTestingIndex = ref(null);

    // 图表输出区域的 HTML
    let plotHTML = ref('');
    let chartUrl = ref('');
    //const chartUrl = ref(backendIP + '/run/cva');

    // 处理列表项的鼠标悬浮事件
    const handleListItemHover = (isHovered, index) => {
      hoveredItem.value = isHovered ? index : null;
    }

    // 训练文件上传
    const handleTrainFileUpload = async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post(API_ROUTES.uploadData, formData, {
          params: { type: "train" }
        });
        if (response.data.message === 'success') {
          trainingFiles.value.push(file.name);
        }
      } catch (error) {
        console.error('上传训练文件时出现错误：', error);
      }
      return false;
    };

    // 测试文件上传
    const handleTestFileUpload = async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post(API_ROUTES.uploadData, formData, {
          params: { type: "test" }
        });
        if (response.data.message === 'success') {
          testingFiles.value.push(file.name);
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
      trainingFiles.value = await fetchFileList('train');
      testingFiles.value = await fetchFileList('test');
    });

    // 删除数据
    const deleteFile = async (fileName, data_type, index) => {
      try {
        await axios.get(API_ROUTES.deleteData, {
          params: { file_name: fileName, type: data_type }
        });
        if (data_type === 'train') {
          trainingFiles.value.splice(index, 1);
        } else if (data_type === 'test') {
          testingFiles.value.splice(index, 1);
        }
      } catch (error) {
        console.error('删除文件时出现错误：', error);
      }
    };

    // 选中训练数据
    const selectTrainingFile = (index) => {
      if (selectedTrainingIndex.value === index) {
        // 已经选中，取消选中
        selectedTrainingIndex.value = null;
      } else {
        // 未选中，选中它
        selectedTrainingIndex.value = index;
      }
    };

    // 选中测试数据
    const selectTestingFile = (index) => {
      if (selectedTestingIndex.value === index) {
        // 已经选中，取消选中
        selectedTestingIndex.value = null;
      } else {
        // 未选中，选中它
        selectedTestingIndex.value = index;
      }
    };


    // 过程监测API请求函数
    const runCVA = async (train_name, test_name) => {
      try {
        const response = await axios.post(API_ROUTES.CVA, {
          train_name: train_name,
          test_name: test_name,
        });
        return response.data;
      } catch (error) {
        console.error('运行CVA时出现错误：', error);
      }
    };


    // 运行过程监测的逻辑
    const runProcess = async () => {
      console.log('runProcess')

      // 在运行过程中，使用选中的训练集索引和测试集索引来获取对应的文件名
      const selectedTrainingFileName = trainingFiles.value[selectedTrainingIndex.value];
      const selectedTestingFileName = testingFiles.value[selectedTestingIndex.value];

      // 执行模型训练和推理操作
      if (selectedOption.value == 'cva') {
        console.log('cva')
        const res = await runCVA(selectedTrainingFileName, selectedTestingFileName);
        console.log(res.html_url)
        chartUrl.value = res.html_url;

      }
      else if (selectedOption.value == 'pca') {
        console.log('pca')
      }
      else {
        console.log('else')
      }

      // 在运行完模型训练和推理后，你可以在 chart-output 和 text-output 区域展示结果
    };

    // 返回需要暴露给模板的数据和方法
    return {
      chartUrl,
      runCVA,
      fetchFileList,
      handleListItemHover,
      hoveredItem,
      selectTestingFile,
      selectTrainingFile,
      selectedTestingIndex,
      selectedTrainingIndex,
      trainingFiles,
      testingFiles,
      handleTrainFileUpload,
      handleTestFileUpload,
      selectedOption,
      runProcess,
      deleteFile,
      plotHTML
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
  flex: 6;
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
  margin-bottom: 20px;
}

.option-select {
  width: 50%;
  margin-right: 20px;
}

.run-button {
  flex-shrink: 0;
}

.chart-output,
.text-output {
  height: 50vh;
  border: 1px solid #ccc;
  background-color: white;
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  padding: 10px;
  overflow: auto;
}

.text-output {
  height: 20vh;
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
