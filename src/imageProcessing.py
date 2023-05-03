import cv2

img = cv2.imread('./images/lena2.jpg')
img_edges=cv2.Canny(img,100,100);
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('./edges_lena.jpg',img_edges)
cv2.imshow('Gray Lena', img_edges)
# Wait for key press and exit if 'q' is pressed
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
