import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Подготовка набора данных с расширением
train_datagen = ImageDataGenerator(
  rescale = 1./255, # Нормализация изображений
  shear_range = 0.2, # Искривление для аугментации
  zoom_range = 0.2, # Увеличение для аугментации
  horizontal_flip = True, # Горизонтальный переворот для аугментации
  validation_split = 0.2 # Резерв 20 % изображений для валидации
)

# Загрузка изображений
train_generator = train_datagen.flow_from_directory(
  'data//train', # Путь к папке с данными
  target_size = (128, 128), # Изменение размера изображений
  batch_size = 32, # Размер партии
  class_mode = 'binary', # Классифицировать, яблоко это или нет
  classes = ['apple', 'not_apple'], # Яблоко и не яблоко
  #subset = 'training' # Использовать для обучения
)

validation_generator = train_datagen.flow_from_directory(
  'data//test', # Путь к папке с данными
  target_size = (128, 128),
  batch_size = 32,
  class_mode = 'binary',
  classes = ['apple', 'not_apple'],
  #subset = 'validation' # Используйте для проверки
)

print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Создание модели CNN
model = Sequential([
  Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 128, 3)), # Сверточный слой
  MaxPooling2D(pool_size = (2, 2)),
# Объединение для уменьшения размера
  Conv2D(64, (3, 3), activation = 'relu'), # Сверточный слой
  MaxPooling2D(pool_size = (2, 2)),
  Conv2D(128, (3, 3), activation = 'relu'),
  MaxPooling2D(pool_size = (2, 2)),
  Flatten(),
# Сплющивание для плотного слоя
  Dense(128, activation = 'relu'), # Плотный слой
  Dropout(0.5), # Отсеивание во избежание чрезмерной подгонки
  Dense(1, activation = 'sigmoid') # Выходной слой с сигмоидом
])

# Компиляция модели
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Отображение краткой информации о модели
model.summary()

# Обучение модели
history = model.fit(
  train_generator,
  #steps_per_epoch = train_generator.samples // train_generator.batch_size,
  validation_data = validation_generator,
  #validation_steps = validation_generator.samples // validation_generator.batch_size,
  epochs = 10
)

# Сохранение модели
model.save('model_apple_cnn.h5')

# Отображение кривых потерь и точности
plt.plot(history.history['accuracy'], label = 'Drive accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Loss of validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()




import cv2
import numpy as np
import matplotlib.pyplot as plt
def detect_red_in_hsv(image_path):
  # Считать изображение из файла
  image = cv2.imread(image_path)

  # Проверьте, правильно ли загружено изображение
  if image is None:
    print(f"Ошибка : не удалось загрузить изображение с {image_path}")
    return
  
  # Преобразуйте изображение из BGR (по умолчанию в OpenCV) в HSV
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Определите цветовые диапазоны для определения красногоц вета в HSV
  lower_red_1 = np.array([0, 50, 50])
  upper_red_1 = np.array([10, 255, 255])
  lower_red_2 = np.array([170, 50, 50])
  upper_red_2 = np.array([180, 255, 255])

  # Создайте маски для двух красных областей
  mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
  mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

  # Объедините две маски
  full_mask = mask1 + mask2

  # Примените маску к исходному изображению
  red_detection = cv2.bitwise_and(image, image, mask = full_mask)
  
  # Преобразуйте изображения из BGR в RGB для отображения с помощью matplotlib
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  red_detection_rgb = cv2.cvtColor(red_detection,
  cv2.COLOR_BGR2RGB)
  
  # Отображение результатов
  plt.figure(figsize = (10, 5))

  plt.subplot(1, 3, 1)
  plt.imshow(image_rgb)
  plt.title("Original image")
  plt.axis("off")

  plt.subplot(1, 3, 2)
  plt.imshow(full_mask, cmap = "gray")
  plt.title("Red detection mask")
  plt.axis("off")

  plt.subplot(1, 3, 3)
  plt.imshow(red_detection_rgb)
  plt.title("Red zones detected")
  plt.axis("off")

  plt.show()

def estimate_distance(image, apple_contour, real_apple_size,focal_length):
  # Получите ограничительную рамку для обнаруженного яблока
  x, y, w, h = cv2.boundingRect(apple_contour)

  # Размер яблока, обнаруженного на изображении (в пикселях)
  apple_image_size = h # Высота яблока на изображении

  # Вычисление расстояния между камерой и яблоком
  distance = (real_apple_size * focal_length) / apple_image_size

  print(f"Расчетное расстояние : {distance:.2f} см")

  return distance


# Использование функции с примером изображения
image = cv2.imread('image/apple3.png')


# Использование функции с изображением яблока
detect_red_in_hsv('image/apple3.png')

