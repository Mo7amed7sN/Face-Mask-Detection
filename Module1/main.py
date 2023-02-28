from imutils.video import FileVideoStream
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import time
import queue
import threading
# from playsound import playsound

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


load_model = keras.models.load_model('densenet121_detection_model.h5')
labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4

cap = VideoCapture(0)

while True:
  try:
    # time.sleep(1)  # simulate time between events
    im = cap.read()
    im = cv2.flip(im, 1, 1)
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    faces = classifier.detectMultiScale(mini)
    for f in faces:
      (x, y, w, h) = [v * size for v in f]
      face_img = im[y:y + h, x:x + w]
      resized = cv2.resize(face_img, (224, 224))
      normalized = resized / 255.0
      reshaped = np.reshape(normalized, (1, 224, 224, 3))
      reshaped = np.vstack([reshaped])
      result = load_model.predict(reshaped)

      if result[0][0] > result[0][1]:
        percent = round(result[0][0] * 100, 2)
      else:
        percent = round(result[0][1] * 100, 2)

      label = np.argmax(result, axis=1)[0]
      if percent < 60.0 and label == 0:
        label = 1
      elif percent < 60.0 and label == 1:
        label = 0

      cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
      cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
      cv2.putText(im, labels_dict[label] + " " + str(percent) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                  (255, 255, 255), 2)

    if im is not None:
      cv2.imshow('COVID Mask Detection Video Feed', im)
    if chr(cv2.waitKey(1) & 255) == 'q':
      break
  except:
    continue
