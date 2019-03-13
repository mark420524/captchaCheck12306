# captchaCheck12306
download the 12306 captcha and check the result
## first step 
  download the captcha from 12306
## second step
  cut the image
## third step
  use CNN to check the captcha. 

  CNN训练数据需要人工识别图片并标记为相应的类型,人工处理验证码的类别进行CNN训练.此步骤比较繁琐

  ```mat_train.py```参考链接
  [https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/ "https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/")

 
  训练依赖 ```keras  sklearn imutils matplotlib tensorflow opencv-python```

  训练命令  
   ```python mat_train.py --dataset traindata  --model aaa.model --labelbin 12306.pickle```
   ```python words_train.py --train -m=DenseNet-BC -k=24 -d=40```

  识别命令 
   ```python classify.py --model pokedex.model --labelbin lb.pickle --image examples/example.png```