import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import cv2
import time

# file_name = "src/tl_detector/light_classification/frozen_inference_graph.pb"
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# =================initialization, only run once =====================
# ====================================================================
file_name = "frozen_inference_graph.pb"
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_name, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# ====================================================================
# ====================================================================

# image is feed
image_np = cv2.imread("70.png")
start_time = time.time()

# =================get_classification()===============================
# ====================================================================
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
image_np_expanded = np.expand_dims(image_np, axis=0)
# image_np_expanded = cv2.resize(image, (800, 600, 3))
(boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
print("==========red============")
print(scores)
print(classes)
print(time.time() - start_time)
# ====================================================================
# ====================================================================
