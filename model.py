import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np


interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_path = 'image.png'  
image = Image.open(image_path).resize((224, 224))
image = image.convert('RGB')
input_image = np.array(image, dtype=np.float32)
input_image = (input_image - 127.5) / 127.5  
interpreter.set_tensor(input_details[0]['index'], [input_image])


interpreter.invoke()


output_data = interpreter.get_tensor(output_details[0]['index'])


print(np.argmax(output_data[0]))
