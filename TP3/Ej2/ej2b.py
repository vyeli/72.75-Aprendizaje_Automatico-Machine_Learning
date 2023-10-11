import cv2
import csv

from utils.svm_library import *
from PIL import Image

csv_output = 'output/dump.csv'
images_folder = "imagenes/"

# imread returns RGB colors

cow_image = cv2.imread(images_folder + "vaca.jpg").reshape(-1, 3)
sky_image = cv2.imread(images_folder + "cielo.jpg").reshape(-1, 3)
grass_image = cv2.imread(images_folder + "pasto.jpg").reshape(-1, 3)

test_image_1 = cv2.imread(images_folder + "gallina.jpg")
image_1_original_shape = test_image_1.shape
test_image_1 = test_image_1.reshape(-1, 3)

test_image_2 = cv2.imread(images_folder + "toro.jpg")
image_2_original_shape = test_image_2.shape
test_image_2 = test_image_2.reshape(-1, 3)

with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['set_1', 'set_2', 'time', 'tp', 'fp', 'fn', 'tn', 'C', 'gamma', 'degree'])

    # Choose parameters
    test_size = 0.2
    kernel = 'linear'
    C_sky_vs_cow = 15
    C_sky_vs_grass = 4
    C_cow_vs_grass = 2

    # Sweet spots
    # Linear: C=5
    # RBF (cow vs grass): C=20
    sky_cow_svm = SVM_train(get_SVM(kernel, C_sky_vs_cow), [sky_image, cow_image])
    sky_grass_svm = SVM_train(get_SVM(kernel, C_sky_vs_grass), [sky_image, grass_image])
    grass_cow_svm = SVM_train(get_SVM(kernel, C_cow_vs_grass), [grass_image, cow_image])

    mask_value = 255
    
    sky_cow_predictions = SVM_predict(sky_cow_svm, test_image_1)
    print(f'Finished predicting sky vs cow')
    sky_grass_predictions = SVM_predict(sky_grass_svm, test_image_1)
    print(f'Finished predicting sky vs grass')
    grass_cow_predictions = SVM_predict(grass_cow_svm, test_image_1)
    print(f'Finished predicting grass vs cow')

    pixels = len(test_image_1)

    for j in np.arange(pixels):
        if sky_cow_predictions[j] == 1 and sky_grass_predictions[j] == 1:
            test_image_1[j][0] = mask_value                 # sky
        elif sky_cow_predictions[j] != 1 and grass_cow_predictions[j] != 1:
            test_image_1[j][2] = mask_value                 # cow
        else:
            test_image_1[j][1] = mask_value                 # grass
    
    cv2.imwrite('output/gallina_masked.jpg', test_image_1.reshape(image_1_original_shape))

    ################################################################################

    sky_cow_predictions = SVM_predict(sky_cow_svm, test_image_2)
    print(f'Finished predicting sky vs cow')
    sky_grass_predictions = SVM_predict(sky_grass_svm, test_image_2)
    print(f'Finished predicting sky vs grass')
    grass_cow_predictions = SVM_predict(grass_cow_svm, test_image_2)
    print(f'Finished predicting grass vs cow')

    pixels = len(test_image_2)

    for j in np.arange(pixels):
        if sky_cow_predictions[j] == 1 and sky_grass_predictions[j] == 1:
            test_image_2[j][0] = mask_value                 # sky
        elif sky_cow_predictions[j] != 1 and grass_cow_predictions[j] != 1:
            test_image_2[j][2] = mask_value                 # cow
        else:
            test_image_2[j][1] = mask_value                 # grass
    
    cv2.imwrite('output/toro_masked.jpg', test_image_2.reshape(image_2_original_shape))