import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_addons as tfa

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

CLASSES = 8
COLORS = ['black', 'red', 'lime', 'blue', 'orange', 'pink', 'cyan', 'magenta']

SAMPLE_SIZE = (256, 256)
OUTPUT_SIZE = (1080, 1920)

def load_images(image, mask):
  image = tf.io.read_file(image)
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize(image, OUTPUT_SIZE)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image /= 255.0

  mask = tf.io.read_file(mask)
  mask = tf.io.decode_png(mask)
  mask = tf.image.rgb_to_grayscale(mask)
  mask = tf.image.resize(mask, OUTPUT_SIZE)
  mask = tf.image.convert_image_dtype(mask, tf.float32)

  masks = []

  for i in range(CLASSES):
    masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

  masks = tf.stack(masks, axis = 2)
  masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

  return image, masks

def augmentate_image(image, masks):
  random_crop = tf.random.uniform((), 0.3, 1)
  image = tf.image.central_crop(image, random_crop)
  masks = tf.image.central_crop(masks, random_crop)

  random_flip = tf.random.uniform((), 0, 1)
  if random_flip >= 0.5:
    # робимо відзеркалення
    image = tf.image.flip_left_right(image)
    masks = tf.image.flip_left_right(masks) # 'masks -' замість 'masks ='

  image = tf.image.resize(image, SAMPLE_SIZE)
  masks = tf.image.resize(masks, SAMPLE_SIZE)

  return image, masks

images = sorted(glob.glob('dataset/images/*.jpg'))
masks = sorted(glob.glob('dataset/masks/*.png'))

images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

dataset = tf.data.Dataset.zip(images_dataset, masks_dataset)

dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.repeat(60)
dataset = dataset.map(augmentate_image, num_parallel_calls=tf.data.AUTOTUNE)

images_and_masks = list(dataset.take(5))
fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize = (15, 5), dpi = 125)

for i, (image, masks) in enumerate(images_and_masks):
  ax[0, i].set_title('Image')
  ax[0, i].set_axis_off()
  ax[0, i].imshow(image)

  ax[1, i].set_title('Mask')
  ax[1, i].set_axis_off()
  ax[1, i].imshow(image / 1.5)

  for channel in range(CLASSES):
    contours = measure.find_contours(np.array(masks[:, :, channel]))
    for contour in contours:
      ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth = 1, color = COLORS[channel])
plt.show()
plt.close()