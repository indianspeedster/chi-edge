import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os
import sys
import time
import argparse

# Navigate to correct position in filesystem
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

# Set up the model
def predict(model, label, image ):
  interpreter = tflite.Interpreter(model_path=model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Prepare and pass the input image  
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image).resize((width, height))
  img = img.convert('RGB')
  input_data = np.array(img)
  input_data = np.expand_dims(img, axis=0)

  start_time = time.time()
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Make a prediction!
  interpreter.invoke()

  # Get and print the result
  output_data = interpreter.get_tensor(output_details[0]['index'])
  inf_time =  time.time() - start_time 
  print(f"time: {inf_time}s" )

  with open(label, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  sorted_result = sorted((e,i) for i,e in enumerate(output_data[0]))
  prediction = sorted_result[-3:][::-1]
  predicted_labels = [" ".join(labels[j[1]].split(" ")[1:]) for j in prediction]
  return prediction, predicted_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    parser.add_argument('--image', help='File path of the image to be recognized.', required=True)
    
    args = parser.parse_args()

    if not args.model or not args.image or not args.label:
        print("Warning: Must provide: model, label, and image file name.")
        sys.exit(1)
    prediction, labels = predict(args.model, args.label, args.image)
    for (confidence, idx), label in zip(prediction, labels):
        print('{:08.6f}: {}'.format(float(confidence / 255.0), label))