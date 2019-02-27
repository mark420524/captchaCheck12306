错误
AttributeError: 'ProgbarLogger' object has no attribute 'log_values'

解决方案
This happens if steps_per_epoch is 0. Make sure that your batch size is not greater than the dataset size to avoid it.

错误 cv2.imread(im_file)返回none

解决 image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8),cv2.IMREAD_COLOR)