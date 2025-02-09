# 如何将文件上传到GitHub

## 创建一个新的GitHub仓库
1. 登录到GitHub。
2. 点击右上角的“+”按钮，然后选择“New repository”。
3. 填写仓库名称和描述，然后点击“Create repository”。

## 在本地创建一个新的Git仓库
1. 打开终端或命令提示符。
2. 导航到你想要存放项目的目录。
3. 运行以下命令初始化一个新的Git仓库：
   ```sh
   git init
   ```

## 将文件添加到仓库中
1. 将你想要上传的文件复制到这个目录中。
2. 运行以下命令将文件添加到Git仓库：
   ```sh
   git add .
   ```

## 提交更改
1. 运行以下命令提交更改：
   ```sh
   git commit -m "Initial commit"
   ```

## 将本地仓库连接到GitHub仓库
1. 运行以下命令将本地仓库连接到GitHub仓库（将`<your-repo-url>`替换为你的GitHub仓库URL）：
   ```sh
   git remote add origin <your-repo-url>
   ```

## 推送更改到GitHub
1. 运行以下命令将更改推送到GitHub：
   ```sh
   git push -u origin master
   ```
