SNR project
===================================

Dependencies
------------

Packages for Arch linux:
```
tensorflow-opt-cuda opencv python-scikit-learn
```

Dataset
------------
Data should be like this (with default path):

```bash
user ~/snrdata $ ls
bounding_boxes.txt  classes.txt  image_class_labels.txt  SET_C
user ~/snrdata $ cat `find . -type f | sort` | md5sum
2d7256832dffd6d06043fc742661e860  -
```

Preparing for fine-tuning pretrained VGG16 model
----------------------------------------------

Downloading checkpoint:
```bash
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
```

Examples
----------------------------------------------
Train 8 layer convolutional network (model is defined in config.py - MyConvConfig)
```bash
./train.py -n Model8Conv -t MyConv
```

Preview results on testset for previously trained VGG16Pretrained model (latest checkpoint)
```bash
./data_preview.py --test --eval -n Model5 -t VGG16Pretrained
```

Train SVM after last hidden layer:
```bash
./trainSVM.py -n Model8Conv -l "Model/batch_normalization_8/batchnorm/add_1:0"
```

Train SVM after softmax:
```bash
./trainSVM.py -n Model8Conv -l "Model/Softmax:0"
```
