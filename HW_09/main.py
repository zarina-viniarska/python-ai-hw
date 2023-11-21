from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave

color_imgs = []
for i in range(1, 6):
  color_imgs.append(Image.open(f'img/color_{i}.jpg'))

# функція для перетворення зображення з колірної моделі rgb в lab
def processed_image(img):
  image = img.resize((256, 256), Image.BILINEAR) # змінимо розмір зображення
  image = np.array(image, dtype = float) # утворюємо матрицю
  size = image.shape
  lab = rgb2lab(1.0 / 255 * image)
  X, Y = lab[:, :, 0], lab[:, :, 1:]
  Y /= 128 # значення в діапазоні [-1; 1]

  X = X.reshape(1, size[0], size[1], 1) # зображення в градації сірого (1 канал)
  Y = Y.reshape(1, size[0], size[1], 2) # кольорове зображення (2 канали)

  return X, Y, size

X, Y, size = processed_image(color_imgs[1])
# X - L -> Lightness,
# Y - (a, b) -> колірні змінні,
# size - розмір

model = Sequential()
model.add(InputLayer(input_shape = (None, None, 1))) # перший вхідний шар

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same')) # матрицею 3 x 3 будемо перебирати пікселі
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same', strides = 2)) # зменшення кроку сканування (не через 1, а через 2)
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', strides = 2))
model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', strides = 2))
model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))

model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding = 'same')) # на останньому шарі функція активації тангенсна, щоб отримати значення [-1; 1]
model.add(UpSampling2D((2, 2)))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x = X, y = Y, batch_size = 1, epochs = 50) # запускаємо процес навчання

def color_img(img):
  X, Y, size = processed_image(img)
  output = model.predict(X)

  output *= 128
  min_vals, max_vals = -128, 127
  ab = np.clip(output[0], min_vals, max_vals)

  curr = np.zeros((size[0], size[1], 3))
  curr[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)
  curr[:, :, 1:] = ab
  return curr

def show_imgs(gray, curr):
  ax_g = plt.subplot(1, 2, 1)
  plt.imshow(gray)
  ax_g.get_xaxis().set_visible(False)
  ax_g.get_yaxis().set_visible(False)

  ax_c = plt.subplot(1, 2, 2)
  plt.imshow(lab2rgb(curr))
  ax_c.get_xaxis().set_visible(False)
  ax_c.get_yaxis().set_visible(False)
  
  plt.show()

for i in range(1, 6):
  gray = Image.open(f'img/gray_{i}.jpg')
  curr = color_img(gray)
  show_imgs(gray, curr)
  # imsave(f'result/img_{i}.jpg', lab2rgb(curr))