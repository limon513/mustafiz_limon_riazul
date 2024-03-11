import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#now load model
cnnmodel = load_model('/home/pegasus/Desktop/Thesis/CustomModel.h5');
tlmodel = load_model('/home/pegasus/Desktop/Thesis/TLModel.h5');

#function to detect pneumonia
def predict_pneumonia(image_path, filename):
   #reading image with opencv and prepreocssing before feeding to model
   image = cv2.imread(os.path.join(image_path,filename))
   #resizeing
   image = cv2.resize(image,(224,224))
   #generalizing
   image = image.astype('float32') / 255.0
   image = np.expand_dims(image, axis=0)
   #predicting the image
   predictioncnn = cnnmodel.predict(image)
   predictiontl = tlmodel.predict(image)

   return predictioncnn, predictiontl ,image

def st_footer(text, links):
    """A simple footer component for Streamlit"""
    footer = "<hr/>"
    footer += text
    footer += '<br/>'
    footer += '\n'.join([f'<a href="{link[0]}">{link[1]}</a>' for link in links])
    st.markdown(footer, unsafe_allow_html=True)


#main function
def main():
   #header
   st.header("Pneumonia Detection Using CNN model", divider='rainbow')
   #uploaded image path
   upimagepath = '/home/pegasus/Desktop/Thesis/up_images/'
    #title and description for the thesis work
   st.subheader('Why we need this tool?')
   st.caption('Pneumonia is an infection that inflames the air sacs in one or both lungs.Early detection of pneumonia plays a critical role in achieving a successful recovery, avoiding serious complications, and minimizing the burden on healthcare systems.')
   st.subheader('Our model can be used to to detect Pneumonia at an early stage with 91\% accuracy.')
   
   #file upload functionality
   uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
   
   if uploaded_file is not None:
      #giving a static name so that it should replace always and upload folder dont get jammed
      filename = 'imgtotest'
      file_extension = os.path.splitext(filename)[1]
      #writing in the folder
      with open(os.path.join(upimagepath,filename),"wb") as file:
         file.write(uploaded_file.getbuffer())
      #if file upload success
      st.success(f"Image '{filename}' uploaded successfully!")
      #getting prediction
      predictioncnn, predictiontl, predicted_image = predict_pneumonia(upimagepath,filename)
      
      #showing the image
      st.image(predicted_image)
      
      # Display the prediction result
      if predictioncnn[0][0] > 0.5:
        st.write("**PredictionCNN:** Pneumonia detected.")
      else:
        st.write("**PredictionCNN:** Pneumonia not detected.")

      if predictiontl[0][0] > 0.5:
        st.write("**PredictionTL:** Pneumonia detected.")
      else:
        st.write("**PredictionTL:** Pneumonia not detected.")

   else:
      st.info("No image uploaded.")
   
   text = "© 2024 - A thesis work by Mustafiz Limon & Riazul"
   links = [
    ("https://github.com/limon513/mustafiz_limon_riazul", "GitHub"),
   ]

   st_footer(text, links)

if __name__ == '__main__':
   main() 