import cv2
import numpy as np
import matplotlib.pyplot as plt

charA = cv2.imread("a.png")

cropped = charA[1:26, 1:26] #selecting the letter A

charAGrayScale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
flat_charA = charAGrayScale.reshape(-1) #flattens the array
#cv2.imshow("Grey", charAGrayScale)
#cv2.waitKey(0)
text = cv2.imread("text.png")
text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)

y_max, x_max = text.shape

mx, my = charAGrayScale.shape

tmp_max = 0
match_x = 0
match_y = 0

for i in range(x_max - mx + 1):
    for j in range(y_max - my + 1):
        cur_selection = text[j: j + my, i: i + mx]
        flat_selection = cur_selection.reshape(-1)
        dot_prod = np.dot(flat_selection, flat_charA)
        mag = np.linalg.norm(flat_selection) * np.linalg.norm(flat_charA)
        if mag != 0:
            evaluation = dot_prod/mag
            if evaluation > tmp_max:
                tmp_max = evaluation
                match_x = i
                match_y = j
print(tmp_max)
print(match_x, match_y)

test_img = text[match_y:match_y+my, match_x:match_x+mx]
cv2.imshow("test",test_img)
cv2.waitKey(0)