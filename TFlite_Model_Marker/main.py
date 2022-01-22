# !pip install -q tflite-model-maker
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
# from tflite_model_maker.config import ExportFormat
# from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

# Tai du lieu
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

# load du lieu tu thu muc ( JEPG,PNG dc sp)
data = DataLoader.from_folder(image_path)

# chia du lieu thanh cac tap train,test,val
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Hiển thị 25 ví dụ hình ảnh với nhãn
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)
  plt.xlabel(data.index_to_label[label.numpy()])
plt.show()

# Tuy chinh mô hình Tensorflow : Mặc định là EfficientNet-Lite0.
model = image_classifier.create(train_data, validation_data = validation_data)
# Library TFlite_model_marker chỉ hỗ trợ EfficientNet-Lite, MobileNetV2, ResNet50 : nếu muốn dùng model khác:
# model = image_classifier.create(train_data, model_spec=model_spec.get('mobilenet_v2'), validation_data=validation_data)
# Nếu muốn dùng model khác model 3 model trên cần khai báo và thay model_spec = inception_v3_spec
# inception_v3_spec = image_classifier.ModelSpec(uri='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
# inception_v3_spec.input_image_shape = [299, 299]

# change values in function CREATE:
'''tflite_model_maker.image_classifier.create(
    train_data, model_spec='efficientnet_lite0', validation_data=None,
    batch_size=None, epochs=None, steps_per_epoch=None, train_whole_model=None,
    dropout_rate=None, learning_rate=None, momentum=None, shuffle=False,
    use_augmentation=False, use_hub_library=True, warmup_steps=None, model_dir=None,
    do_train=True
)'''

# xem cấu trúc của mô hình
model.summary()

# đánh giá mô hình trên dữ liệu từ tệp test
loss, accuracy = model.evaluate(test_data)

# Dự đoán và hiển thị kết quả của 100 ảnh thử nghiệm
# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

# Then plot 100 test images and their predicted labels.
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.
plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)

  predict_label = predicts[i][0][0]
  color = get_label_color(predict_label,
                          test_data.index_to_label[label.numpy()])
  ax.xaxis.label.set_color(color)
  plt.xlabel('Predicted: %s' % predict_label)
plt.show()

# Xuất sang mô hình Tensorflow lite.
model.export(export_dir='.')