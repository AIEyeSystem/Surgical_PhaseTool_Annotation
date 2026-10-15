# 使用sam3文本提示功能，自动标注图片。并保存为IVAS格式标注文件

## Checkpoint
需要自行下载sam.pt文件，下载页面：https://huggingface.co/facebook/sam3/tree/main 

## USE
```shell
python .\main.py -c .\sam3.pt -p person cup table -i .\test_images
```
