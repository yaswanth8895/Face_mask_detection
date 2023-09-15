#load tflite model and run inference on a single image
import numpy as np
import tensorflow as tf
import cv2

#load model
interpreter = tf.lite.Interpreter(model_path="cnn2.tflite")
interpreter.allocate_tensors()
# Take image input from camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #resize image
    img = cv2.resize(frame, (224, 224))
    #convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #convert to numpy array
    img = np.asarray(img)
    #add batch dimension
    img = np.expand_dims(img, axis=0)
    #normalize image
    img = img / 255.0
    #convert to float32
    img = img.astype(np.float32)
    #set input tensor
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img)
    #run inference
    interpreter.invoke()
    #get output tensor
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    #get class with highest probability
    prediction = output[0][0]
    #print prediction
    if prediction > 0.55:
        cv2.putText(frame, "Mask", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #show image
    cv2.imshow("image", frame)
    #wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break