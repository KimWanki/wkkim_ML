# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.mobilenetv2 import mobilenet_v2, preprocess_input
model = mobilenet_v2(include_top=False, weights='imagenet',input_shape=(224,224,3))


#false 로 할 경우, 마지막
for layer in model.layers:
    layer.trainable = True


from keras.layers import Dense, Dropout, GlobalAveragePooling2D
x = model.output
x = GlobalAveragePooling2D()(x) #tf와 양식이 동일.
x = Dropout(0.5)(x)


from keras.models import Model
#dense : 출력 class를 줄인다!
predictions = Dense(3, activation= 'softmax')(x)
model = Model(inputs = model.input, output = predictions)


from keras.preprocessing.image import ImageDataGenerator
val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

#directory 별로 만들겠다
val_generator = val_datagen.flow_from_directory(
    directory = '/Users/kimwanki/developer/testcase/data/validation',
    target_size = (224, 224),
    batch_size = 10,
    #one hot으로 만들어줌
    class_mode = 'categorical',
    #binary - > 인풋값 그대로.
    shuffle=False
)
import math,os
folder_path = '/Users/kimwanki/developer/testcase/data/validation'
folder = os.listdir(folder_path)
count = 0
for i in folder:
    count += len(os.listdir(folder_path+'/'+i))

model.load_weights('/Users/kimwanki/developer/testcase/data/model.h5')
preds = model.predict_generator(val_generator,verbose=1,steps=math.ceil(count/10))

#preds = tf code result

print(preds)
