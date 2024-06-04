import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import packages

def main():
    st.set_page_config(
        page_title='Cifar10 Web Classifier'
    )
    st.title("Cifar10 Web Classifier")
    st.write('Upload any images to see if classification is correct')
    
    file = st.file_uploader('Please upload an image', type=[
        'jpg',
        'png'
    ])
    if file:
        image = Image.open(file)
        st.divider()
        st.subheader('View Image')
        st.image(file, use_column_width=True, clamp=True, caption=file.name)        
        resized_image = image.resize((32, 32))
        image_array = np.array(resized_image) / 255
        image_array = image_array.reshape((1, 32, 32, 3))
        
        with st.spinner('Preparing Model and Predicting the image'):
            model = packages.model(packages.MODEL_FULLPATH)
            predictions = model.predict(image_array)
        
        fig, ax = plt.subplots()
        y_pos = np.arange(len(packages.CIFAR10_CLASSES))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(map(lambda string: string.capitalize(), packages.CIFAR10_CLASSES))
        ax.invert_yaxis()
        ax.set_xlabel("Probability".capitalize())
        ax.set_title('Cifar10 Predictions')
        
        st.divider()
        st.subheader("View Graph")
        st.pyplot(fig, clear_figure=True)
        st.snow()
    else:
        st.warning('You have not uploaded an image yet.')
        
if __name__ == '__main__':
    main()