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

# =========================== HW_11 ===========================
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

# =========================== новий код ===========================

train_dataset = dataset.take(2000).cache()
test_dataset = dataset.skip(2000).take(100).cache()

train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

def inp_layer():
  return tf.keras.layers.Input(shape = SAMPLE_SIZE + (3,))

def downsamp_block(filters, size, batch_norm = True):
  initializer = tf.keras.initializers.GlorotNormal()

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size,
                             strides = 2,
                             padding = 'same',
                             kernel_initializer = initializer,
                             use_bias = False)
  )
  if batch_norm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample_block(filters, size, dropout = False):
  initializer = tf.keras.initializers.GlorotNormal()

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size,
                                      strides = 2,
                                      padding = 'same',
                                      kernel_initializer = initializer,
                                      use_bias = False)
  )
  result.add(tf.keras.layers.BatchNormalization())
  if dropout:
    result.add(tf.keras.layers.Dropout(0.25))
  result.add(tf.keras.layers.ReLU())

  return result

def outp_layer(size):
  initializer = tf.keras.initializers.GlorotNormal()
  return tf.keras.layers.Conv2DTranspose(CLASSES, size,
                                         strides = 2,
                                         padding = 'same',
                                         kernel_initializer = initializer,
                                         activation = 'sigmoid')

inp_layer = inp_layer()
downsample_stack = [
    downsamp_block(64, 4, batch_norm = False),
    downsamp_block(128, 4),
    downsamp_block(256, 4),
    downsamp_block(512, 4),
    downsamp_block(512, 4),
    downsamp_block(512, 4),
    downsamp_block(512, 4)
]
upsample_stack = [
    upsample_block(512, 4, dropout = True),
    upsample_block(512, 4, dropout = True),
    upsample_block(512, 4, dropout = True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]
outp_layer = outp_layer(4)

x = inp_layer
downsample_skips = []

for block in downsample_stack:
  x = block(x)
  downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
  x = up_block(x)
  x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = outp_layer(x)

# створення моделі
unet_like = tf.keras.Model(inputs = inp_layer, outputs = out_layer)
tf.keras.utils.plot_model(unet_like, show_shapes = True, dpi = 72)

def dice_mc_metric(a, b):
  a = tf.unstack(a, axis = 3)
  b = tf.unstack(b, axis = 3)

  dice_sum = 0

  for i, (aa, bb) in enumerate(zip(a, b)):
    numerator = 2 * tf.math.reduce_sum(aa * bb) + 1
    denumerator = tf.math.reduce_sum(aa + bb) + 1
    dice_sum += numerator / denumerator

  avg_dice = dice_sum / CLASSES

  return avg_dice

def dice_mc_loss(a, b):
  return 1 - dice_mc_metric(a, b)

def dice_bce_mc_loss(a, b):
  return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

unet_like.compile(optimizer = 'adam', loss = [dice_bce_mc_loss], metrics = [dice_mc_metric])

# навчання мережі
history_dice = unet_like.fit(train_dataset, validation_data = test_dataset, epochs = 25, initial_epoch = 0)
# збереження ваг мережі у файл
unet_like.save_weights('networks/unet_like')

import os
# завантаження мережі з файлу
unet_like.load_weights('networks/unet_like')
# ============= тестування мережі =============
rgb_colors = [
    (0,   0,   0),
    (255, 0,   0),
    (0,   255, 0),
    (0,   0,   255),
    (255, 165, 0),
    (255, 192, 203),
    (0,   255, 255),
    (255, 0,   255)
]

frames = sorted(glob.glob('videos/original_video/*.jpg'))

for filename in frames:
    frame = imread(filename)
    sample = resize(frame, SAMPLE_SIZE)

    predict = unet_like.predict(sample.reshape((1,) +  SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))

    scale = frame.shape[0] / SAMPLE_SIZE[0], frame.shape[1] / SAMPLE_SIZE[1]

    frame = (frame / 1.5).astype(np.uint8)

    for channel in range(1, CLASSES):
        contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
        contours = measure.find_contours(np.array(predict[:,:,channel]))

        try:
            for contour in contours:
                rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                           contour[:, 1] * scale[1],
                                           shape=contour_overlay.shape)

                contour_overlay[rr, cc] = 1

            contour_overlay = dilation(contour_overlay, disk(1))
            frame[contour_overlay == 1] = rgb_colors[channel]
        except:
            pass

    imsave(f'videos/processed/{os.path.basename(filename)}', frame)