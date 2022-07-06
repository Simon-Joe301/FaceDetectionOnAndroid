import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

# 调用摄像头摄像头
cap = cv2.VideoCapture(0) #opencv的函数，参数0代表调用电脑自带摄像，若改为1则调用外设摄像

while(True):
    # 获取摄像头拍摄到的画面
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    img = frame
    for (x,y,w,h) in faces:
        # 画出人脸框，蓝色，画笔宽度微
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_area = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(face_area)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

    # 实时展示效果画面
    cv2.imshow('frame2',img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 最后，关闭所有窗口qqqq
cap.release()
cv2.destroyAllWindows()
