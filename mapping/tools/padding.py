import cv2
img = cv2.imread('court-plan/tennis-court-plan.jpg')
replicate = cv2.copyMakeBorder(img,10,10,100,100,cv2.BORDER_REPLICATE)
cv2.imwrite('court-plan/tennis-court-plan-padded.png', replicate)