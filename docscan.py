# Packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# Construction du parseur d'arguments et parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Chemin de l'image à scanner")
args = vars(ap.parse_args())

# Chargement de l'image, calcul du ratio de redimensionnement et redimensionnement
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# Conversion de l'image en niveaux de gris, floutage, détection des bords
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Affichage de l'image originale et de l'image après détection des bords
print("Etape 1: Détection des bords")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Trouver les contours dans l'image après détection des bords, puis initialiser
# la détection du contour du document
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# Boucle sur les contours
for c in cnts:
    # Approximation du contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # Si notre contour a quatre points, alors nous pouvons supposer que nous avons trouvé le document
    if len(approx) == 4:
        screenCnt = approx
        break

# Affichage du contour du document (étape 2)
print("Etape 2: Détection du contour du document")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Appliquer four_point_transform pour obtenir une ortophoto
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Conversion de l'image en niveaux de gris, puis en noir et blanc
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# Affichage de l'image scannée
print("Etape 3: Application de la perspective")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
