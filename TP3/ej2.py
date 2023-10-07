import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn as sk

images_folder = "imagenes_2/"

# 2.a
# imread returns RGB colors
cow_image = cv2.imread(images_folder + "vaca.jpg")
sky_image = cv2.imread(images_folder + "cielo.jpg")
grass_image = cv2.imread(images_folder + "pasto.jpg")

# 2.b
# Test train split

test_size = 0.2

cow_train, cow_test = sk.model_selection.train_test_split(cow_image, test_size=test_size)
sky_train, sky_test = sk.model_selection.train_test_split(sky_image, test_size=test_size)
grass_train, grass_test = sk.model_selection.train_test_split(grass_image, test_size=test_size)
