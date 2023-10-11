import cv2
import csv

from utils.svm_library import *

csv_output = 'output/mixed_kernel.csv'
images_folder = "imagenes/"

# imread returns RGB colors
cow_image = cv2.imread(images_folder + "vaca.jpg").reshape(-1, 3)
sky_image = cv2.imread(images_folder + "cielo.jpg").reshape(-1, 3)
grass_image = cv2.imread(images_folder + "pasto.jpg").reshape(-1, 3)

with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['set_1', 'set_2', 'time', 'tp', 'fp', 'fn', 'tn', 'C', 'gamma', 'degree'])

    # Choose parameters
    test_size = 0.2
    kernel = 'linear'
    C_sky_vs_cow = 15
    C_sky_vs_grass = 4
    C_cow_vs_grass = 2
    runs = 10

    # Sweet spots
    # Linear: C=5
    # RBF (cow vs grass): C=20
    for i in range(runs):
        _svm = get_SVM('linear', C_sky_vs_cow)
        run_SVM(_svm, writer, [sky_image, cow_image], ['sky', 'cow'], test_size, C_sky_vs_cow)

        _svm = get_SVM('linear', C_sky_vs_grass)
        run_SVM(_svm, writer, [sky_image, grass_image], ['sky', 'grass'], test_size, C_sky_vs_grass)

        _svm = get_SVM('linear', C_cow_vs_grass)
        run_SVM(_svm, writer, [grass_image, cow_image], ['grass', 'cow'], test_size, C_cow_vs_grass)
    print(f'\t\tRun {i + 1} of total {runs} finished')