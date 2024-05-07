import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from PIL import Image

# Загрузка модели
loaded_model = tf.keras.models.load_model('mymodel.h5', compile = False)
loaded_model.compile(Adamax(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Загружаем картинку для предсказания
image_path = 'test.png'
image = Image.open(image_path)

# Подготавливаем картинку
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Делаем предсказание
predictions = loaded_model.predict(img_array)
class_labels = classes
score = tf.nn.softmax(predictions[0])

print(f"{class_labels[tf.argmax(score)]}")
