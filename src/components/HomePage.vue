<template>
  <h2>Process Monitoring Tool</h2>
  <div>
    <div class="workspace-container">
      <!-- 训练数据 -->
      <div class="data-card">
        <div>
          <h3>训练数据</h3>
        </div>
        <div style="margin-top: 10px;margin-bottom: 10px;">
          <a-upload-dragger :multiple="true" action="#" :before-upload="handleTrainFileUpload" accept=".csv"
            :show-upload-list="false">
            <p class="ant-upload-drag-icon">
              <upload-outlined></upload-outlined>
            </p>
            <p class="ant-upload-text" style="font-size: small;">单击或拖动文件到此区域进行上传</p>
            <p class="ant-upload-hint" style="font-size: small;">
              支持单个或批量上传。
            </p>
          </a-upload-dragger>
        </div>

        <a-list>
          <!-- 添加鼠标悬浮事件处理和垃圾桶图标 -->
          <a-list-item v-for="(fileName, index) in trainingFiles" :key="index" @click="selectTrainingFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedTrainingIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <span class="check-icon" v-if="selectedTrainingIndex === index">
              <CheckSquareOutlined />
            </span>
            {{ fileName }}
            <span class="trash-button" v-if="hoveredItem === index" type="delete" @click.stop="deleteTrainingFile(index)">
              <DeleteOutlined />
            </span>
          </a-list-item>
        </a-list>
      </div>

      <!-- 测试数据 -->
      <div class="data-card">
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
              支持单个或批量上传。
            </p>
          </a-upload-dragger>
        </div>
        <a-list>
          <!-- 添加鼠标悬浮事件处理和垃圾桶图标 -->
          <a-list-item v-for="(fileName, index) in testingFiles" :key="index" @click="selectTestingFile(index)"
            :class="{ 'list-item-content': true, 'highlighted': selectedTestingIndex === index }"
            @mouseenter="handleListItemHover(true, index)" @mouseleave="handleListItemHover(false, index)">
            <div>
              <span class="check-icon" v-if="selectedTestingIndex === index">
                <CheckSquareOutlined />
              </span>
              {{ fileName }}
            </div>
            <span class="trash-button" v-if="hoveredItem === index" type="delete" @click.stop="deleteTestingFile(index)">
              <DeleteOutlined />
            </span>

          </a-list-item>
        </a-list>
      </div>

      <!-- 算法选择和运行 -->
      <div class="process-card">
        <div class="process-controls">
          选择算法：
          <a-select v-model="selectedOption" class="option-select" placeholder="PCA">
            <a-select-option value="option1">PCA</a-select-option>
            <a-select-option value="option2">CAE</a-select-option>
            <!-- 添加更多选项 -->
          </a-select>
          <a-button type="primary" @click="runProcess" class="run-button">运行</a-button>
        </div>

        <div class="chart-output">
          <!-- 图表输出区域 -->
        </div>

        <div class="text-output">
          <!-- 文本输出区域 -->
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Button, UploadDragger } from 'ant-design-vue';
import { List } from 'ant-design-vue';
import { ref } from 'vue';
import { UploadOutlined, CheckSquareOutlined, DeleteOutlined } from '@ant-design/icons-vue';

export default {
  name: 'HomePage',
  components: {
    AUploadDragger: UploadDragger,
    AButton: Button,
    AList: List,
    AListItem: List.Item,
    UploadOutlined, CheckSquareOutlined, DeleteOutlined
  },
  setup() {
    // 算法选择
    const selectedOption = ref(null);

    // 文件上传相关
    const trainingFiles = ref([]);
    const testingFiles = ref([]);

    // 是否悬浮【a-list-item 并没有内置用于判断鼠标是否悬浮的 API，所以利用mouseenter等实现】
    const hoveredItem = ref(null);

    // 点击选中的训练集和测试集索引
    const selectedTrainingIndex = ref(null);
    const selectedTestingIndex = ref(null);


    // 处理列表项的鼠标悬浮事件
    const handleListItemHover = (isHovered, index) => {
      hoveredItem.value = isHovered ? index : null;
    }

    // 训练文件上传
    const handleTrainFileUpload = (file) => {
      // 获取文件名并存储
      const fileName = file.name;
      if (fileName) {
        trainingFiles.value.push(fileName);
      }
      return false;
    };
    // 测试文件上传
    const handleTestFileUpload = (file) => {
      // 获取文件名并存储
      const fileName = file.name;
      if (fileName) {
        testingFiles.value.push(fileName);
      }
      return false;
    };


    // 删除数据
    const deleteTrainingFile = (index) => {
      trainingFiles.value.splice(index, 1);
    };
    const deleteTestingFile = (index) => {
      testingFiles.value.splice(index, 1);
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


    // 运行过程监测的逻辑
    const runProcess = () => {

      // 在运行过程中，使用选中的训练集索引和测试集索引来获取对应的文件名
      const selectedTrainingFileName = trainingFiles.value[selectedTrainingIndex.value];
      const selectedTestingFileName = testingFiles.value[selectedTestingIndex.value];

      // 执行模型训练和推理操作
      // 使用 selectedTrainingFileName 和 selectedTestingFileName 进行相应操作
      console.log(selectedTrainingFileName)
      console.log(selectedTestingFileName)

      // 在运行完模型训练和推理后，你可以在 chart-output 和 text-output 区域展示结果
    };

    // 返回需要暴露给模板的数据和方法
    return {
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
      deleteTrainingFile,
      deleteTestingFile
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
