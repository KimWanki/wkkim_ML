import tensorflow as tf

# saved_model_dir = '/Users/kimwanki/PycharmProjects/untitled3/tflite/tf/model.pb'
# converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open('/Users/kimwanki/PycharmProjects/untitled3/tflite/tf/converted_model.tflite', 'wb').write(tflite_model)

# model = '/Users/kimwanki/PycharmProjects/untitled3/tflite/model.pb'
# # Convert the model.
# converter = tf.lite.TFLiteConverter.from_saved_model(model)
# tflite_model = converter.convert()
#
# # Save the TF Lite model.
# with tf.io.gfile.GFile('model.tflite', 'wb') as f:
#   f.write(tflite_model)

import tensorflow as tf

import tensorflow as tf
model=tf.keras.models.load_model("/Users/kimwanki/PycharmProjects/untitled3/tflite/model.pb")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("'/Users/kimwanki/PycharmProjects/untitled3/tflite/converted_model.tflite", "wb").write(tflite_model)

# # # saved_model_dir = '/Users/kimwanki/PycharmProjects/untitled3/tflite/model.pb'
# # # converter = tf.lite.TFLiteConverter.from_keras_model(saved_model_dir)
# # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
# #                                        tf.lite.OpsSet.SELECT_TF_OPS]
# # tflite_model = converter.convert()
#
# open('/Users/kimwanki/PycharmProjects/untitled3/tflite/converted_model.tflite', 'wb').write(tflite_model)
