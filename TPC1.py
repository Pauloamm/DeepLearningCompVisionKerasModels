import tensorflow.keras as tf_keras
import PIL.Image as PImage
import matplotlib.pyplot as plt
import numpy as np


def LoadImageLinks():

    linkList = []
    linkList.append("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/SL_Bundala_NP_asv2020-01_img08.jpg/800px-SL_Bundala_NP_asv2020-01_img08.jpg?20200430234629")#bird
    linkList.append('https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Strawberry_ice_cream_cone_%285076899310%29.jpg/640px-Strawberry_ice_cream_cone_%285076899310%29.jpg')#ice cream
    linkList.append('https://st.depositphotos.com/1979329/2126/v/950/depositphotos_21261765-stock-illustration-cellphone.jpg')#cellphone
    linkList.append('https://www.receitas-sem-fronteiras.com/media/6118720-cookie-de-chocolate-chip_crop.jpg/rh/cookies-tipo-americano.jpg')#cookie
    linkList.append('https://m.media-amazon.com/images/I/61Jigwd1kKL._AC_SL1500_.jpg')#football ball
    linkList.append('https://media.istockphoto.com/photos/basketball-picture-id170096587?k=20&m=170096587&s=612x612&w=0&h=Umu6ELi7aPSpCPE7hMPKWVYZUoRfdNek2ieBI5RrCCs=')#basketball ball
    linkList.append('https://carwow-uk-wp-3.imgix.net/Volvo-XC40-white-scaled.jpg')#car
    linkList.append('https://m.media-amazon.com/images/I/41tCIsGV8UL.jpg')#chair
    linkList.append('https://d1aeri3ty3izns.cloudfront.net/media/23/235459/600/preview_4.jpg')#guitar
    linkList.append('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Weizenbier.jpg/640px-Weizenbier.jpg')#beer

    return linkList

def xCeptionModel(linkList,imagenet_labels):

    model = tf_keras.applications.xception.Xception(weights="imagenet")

    imageRes = 299
    counter = 0

    showImages = []
    processedImages = []
    for link in linkList:
        image_path = tf_keras.utils.get_file('image{}.jpg'.format(counter), link)
        img = tf_keras.preprocessing.image.load_img(image_path).resize((imageRes, imageRes))
        showImages.append(img)

        x = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf_keras.applications.xception.preprocess_input(x)
        processedImages.append(x)
        counter += 1

    counter = 0
    for image in processedImages:
        result = model.predict(image)
        for r in result:
            predicted_class = np.argmax(r, axis=-1)
            predicted_class_name = imagenet_labels[predicted_class + 1]

            plt.imshow(showImages[counter])
            plt.axis('off')
            _ = plt.title("Prediction: " + predicted_class_name.title())
            plt.show()

        counter += 1

def ResNet50Model(linkList,imagenet_labels):

    model = tf_keras.applications.resnet50.ResNet50(weights="imagenet")
    imageRes = 224
    counter = 0

    showImages = []
    processedImages = []
    for link in linkList:
        image_path = tf_keras.utils.get_file('image{}.jpg'.format(counter), link)
        img = tf_keras.preprocessing.image.load_img(image_path).resize((imageRes,imageRes))
        showImages.append(img)

        x = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf_keras.applications.resnet50.preprocess_input(x)
        processedImages.append(x)
        counter+=1


    counter = 0
    for image in processedImages:
        result = model.predict(image)
        for r in result:
            predicted_class = np.argmax(r, axis=-1)
            predicted_class_name = imagenet_labels[predicted_class + 1]

            plt.imshow(showImages[counter])
            plt.axis('off')
            _ = plt.title("Prediction: " + predicted_class_name.title())
            plt.show()

        counter += 1

def ResNet152Model(linkList,imagenet_labels):

    model = tf_keras.applications.resnet.ResNet152(weights="imagenet")

    imageRes = 224
    counter = 0

    showImages = []
    processedImages = []
    for link in linkList:
        image_path = tf_keras.utils.get_file('image{}.jpg'.format(counter), link)
        img = tf_keras.preprocessing.image.load_img(image_path).resize((imageRes, imageRes))
        showImages.append(img)

        x = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf_keras.applications.resnet.preprocess_input(x)
        processedImages.append(x)
        counter += 1

    counter = 0
    for image in processedImages:
        result = model.predict(image)
        for r in result:
            predicted_class = np.argmax(r, axis=-1)
            predicted_class_name = imagenet_labels[predicted_class + 1]

            plt.imshow(showImages[counter])
            plt.axis('off')
            _ = plt.title("Prediction: " + predicted_class_name.title())
            plt.show()

        counter += 1

def VGG16Model(linkList,imagenet_labels):

    model = tf_keras.applications.vgg16.VGG16(weights="imagenet")
    imageRes = 224

    counter = 0
    showImages = []
    processedImages = []
    for link in linkList:
        image_path = tf_keras.utils.get_file('image{}.jpg'.format(counter), link)
        img = tf_keras.preprocessing.image.load_img(image_path).resize((imageRes, imageRes))
        showImages.append(img)

        x = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf_keras.applications.vgg16.preprocess_input(x)
        processedImages.append(x)
        counter += 1

    counter = 0
    for image in processedImages:
        result = model.predict(image)
        for r in result:
            predicted_class = np.argmax(r, axis=-1)
            predicted_class_name = imagenet_labels[predicted_class + 1]

            plt.imshow(showImages[counter])
            plt.axis('off')
            _ = plt.title("Prediction: " + predicted_class_name.title())
            plt.show()

        counter += 1

def VGG19Model(linkList,imagenet_labels):

    model = tf_keras.applications.vgg19.VGG19(weights="imagenet")

    imageRes = 224

    counter = 0
    showImages = []
    processedImages = []
    for link in linkList:
        image_path = tf_keras.utils.get_file('image{}.jpg'.format(counter), link)
        img = tf_keras.preprocessing.image.load_img(image_path).resize((imageRes, imageRes))
        showImages.append(img)

        x = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf_keras.applications.vgg19.preprocess_input(x)
        processedImages.append(x)
        counter += 1

    counter = 0
    for image in processedImages:
        result = model.predict(image)
        for r in result:
            predicted_class = np.argmax(r, axis=-1)
            predicted_class_name = imagenet_labels[predicted_class + 1]

            plt.imshow(showImages[counter])
            plt.axis('off')
            _ = plt.title("Prediction: " + predicted_class_name.title())
            plt.show()

        counter += 1



if __name__ == '__main__':

    labels_path = tf_keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    imageLinks = LoadImageLinks()


    #WORKING MODELS
    #ResNet50Model(imageLinks, imagenet_labels)
    #ResNet152Model(imageLinks,imagenet_labels)
    #VGG16Model(imageLinks,imagenet_labels)
    #VGG19Model(imageLinks,imagenet_labels)
    xCeptionModel(imageLinks,imagenet_labels)


