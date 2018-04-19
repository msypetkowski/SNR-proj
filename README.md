SNR project
===================================

Dependencies
------------

###Arch linux packages:
opencv

Dataset
------------
Data should be like this (with default path):

```bash
user ~/snrdata $ ls
bounding_boxes.txt  classes.txt  image_class_labels.txt  SET_C
user ~/snrdata $ cat `find . -type f | sort` | md5sum
2d7256832dffd6d06043fc742661e860  -
```
