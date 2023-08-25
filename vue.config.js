const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  pluginOptions: {
    electronBuilder: {
      chainWebpackMainProcess: (config) => {
        config.output.filename((file) => {
                  if (file.chunk.name === 'index') {
                      return 'background.js';
                  } else {
                      return '[name].js';
                  }
              });
      }
    }
  }
})
