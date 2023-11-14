import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import json

plt.rcParams.update({'font.size': 8})
plt.subplots_adjust(hspace = 0.5)

model = keras.applications.VGG16()

with open('img_labels.json', 'r') as file:
    labels = json.load(file)

for i in range(1, 11):
  img = Image.open(f'img/{i}.jpg')
  
  img = np.array(img)
  x = keras.applications.vgg16.preprocess_input(img)
  x = np.expand_dims(x, axis = 0)
  res = model.predict(x)
  img_title = labels[str(np.argmax(res))]
  
  plt.subplot(3, 4, i)
  plt.xticks([])
  plt.yticks([])
  
  if len(img_title) > 16:
    plt.title(f"{img_title[:16]}..")
  else:
    plt.title(img_title)

  plt.imshow(img)