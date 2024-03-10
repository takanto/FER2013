import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_augmentations(X, y, df, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, 
                       brightness_range=(0.95, 1.05), horizontal_flip=True, vertical_flip=True, fill_mode='nearest', target_count=None):
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=rotation_range,
      width_shift_range=width_shift_range,
      height_shift_range=height_shift_range,
      zoom_range=zoom_range,
      brightness_range=brightness_range,
      horizontal_flip=horizontal_flip,
      vertical_flip=vertical_flip, 
      fill_mode=fill_mode
  )

  balanced_X = []
  balanced_y = []

  target_count = target_count if target_count else max(df['emotion'].value_counts())

  distribution = []

  for class_label in df['emotion'].unique():
      class_indices = np.where(y[:, class_label] == 1)[0]
      class_images = X[class_indices]
      class_labels = y[class_indices]
      num_images = class_images.shape[0]
      
      augmentations_needed = target_count - num_images
      
      while True:
          for img, label in zip(class_images, class_labels):
              if (augmentations_needed <= 0):
                break
              img = img.reshape((1,) + img.shape)
              label = label.reshape((1,) + label.shape)
              augmented_img = next(datagen.flow(img))
              balanced_X.append(augmented_img.squeeze())
              balanced_y.append(label.squeeze())
              augmentations_needed -= 1
          if (augmentations_needed <= 0):
                break

      balanced_X.extend(class_images)
      balanced_y.extend(class_labels)

  balanced_X = balanced_X[:target_count]
  balanced_y = balanced_y[:target_count]
  balanced_X = np.array(balanced_X)
  balanced_y = np.array(balanced_y)

  for class_label in df['emotion'].unique():
      class_indices = np.where(balanced_y[:, class_label] == 1)[0]
      class_images = balanced_X[class_indices]
      class_labels = balanced_y[class_indices]
      num_images = class_images.shape[0]
      distribution.append(num_images)

  return balanced_X, balanced_y, distribution

def plot_rand_imgs(data, img_size, pred='nan', actual='nan'):
  # images selected
  random_ids = np.random.choice(data.shape[0], 9)

  # for every image, plot it as a subplot
  j = 0
  fig = plt.figure()
  for i in random_ids:
    ax = plt.subplot(330 + 1 + j)
    ax.axis('off')
    # if there is a list of predictions, display it as title
    if type(actual) != str:
      title = f'actual: {actual[i]}'
      if type(pred) != str:
        title += f', pred: {pred[i]}'
      ax.title.set_text(title)
    # if RGB is stored, convert it to RGB
    img = data[i]
    if img_size[2]==3:
      ax.imshow(cv2.cvtColor(img.reshape(img_size[0], img_size[1], 3), cv2.COLOR_BGR2RGB))
    # if grayscale is stored, plot it with gray color map
    elif img_size[2]==1:
      ax.imshow(img.reshape(img_size[0], img_size[1], 1), plt.get_cmap('gray'))

    j += 1

  plt.axis('off')
  plt.show()

def distribution_plot(label_counts):
  emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

  # Plot the bar chart
  plt.figure(figsize=(6, 4))
  plt.bar(emotion_labels, label_counts, color='skyblue')
  plt.title('Distribution of Emotion Labels')
  plt.xlabel('Emotion Labels')
  plt.ylabel('Counts')
  plt.xticks(rotation=45)
  plt.show()