# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 01:03:24 2019

@author: Ali Cherif
"""

#Creating the model
def create_model():
    #Initialising the CNN 
    model = models.Sequential()
    
    #Step1 - Convolution
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    
    #Step2 - Pooling
    model.add(layers.MaxPooling2D((2, 2)))
    
    #Step3 - Adding other convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    
    #Step3 - Flattening
    model.add(layers.Flatten())
    
    #Step 4 - Full connection
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    #Compiling CNN
    model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    
    #returns the model
    return model
    
    
#To visualize the differnet layers and parameters of the model    
def visualize_model(model):
    model.summary()


#Load parameters from an existing model in the defined path
def load_model(model, path):
    model.load_weights(path)



#returns a prediction of an image using url
def predict_from_url(model, url):
    #importing libraries and packages
    import requests
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import load_img
    import numpy as np
    
    #download the file in a local directory
    r = requests.get(url, allow_redirects=True)
    open('./Downloaded_file/file', 'wb').write(r.content)
    
    #transform the downloaded file(image) into an array
    image = img_to_array(load_img('./Downloaded_file/file', target_size=(128,128))) / 255
    
    #add ad additional dimension to the array to fit the model
    image = np.expand_dims(image, axis=0)
    
    #get the prediction
    prediction = model.predict_on_batch(image)
    
    #return the result
    return prediction



if __name__ == "__main__":
    #Importing packages and libraries
    from tensorflow.keras import layers, models
    import sys
    
    #creating the model
    model = create_model()
    
    #load an existing model
    path = './Saved_model/my_checkpoint'
    load_model(model, path)
    
    #visualize the model
    visualize_model(model)
    
    #predict the classification of an image from url
    url = str(sys.argv[1])
    prediction = predict_from_url(model, url)
    
    #show the result
    prc = prediction[0][0] * 100
    prc = round(prc, 2)
    
    if (prediction > 0.5):
        print('This image is ' + str(prc) + '% inappropriate')
    else:
        prc = 100 - prc
        print('This image is ' + str(prc) + '% clean')