import numpy as np
import cv2 as cv
import time

#write text to frame
def warning_text(frame,text):
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    frame_text = cv.putText(frame, text, org, font, 
                fontScale, color, thickness, cv.LINE_AA)
    return frame_text

#cap = cv.VideoCapture(0)
cap = cv.VideoCapture("rtsp://admin:qwerty11@skysys.iptime.org:8554/Streaming/Channels/201")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
fps = cap.get(cv.CAP_PROP_FPS)
i=0
while True:
    i=i+1
    ret, frame = cap.read()
    # if i%(3*fps) ==0:
    #         frame = warning_text(frame,'FIRE')
    #         cv.imshow('frame', frame)
    #         time.sleep(1)

    if (i+1)%(3*fps) ==0:
            frame = warning_text(frame,'CHECKING')
    elif i%(3*fps) ==0:
            time.sleep(1)
    else :
         frame = warning_text(frame,'NO_CHECK')


    # if frame is read correctly ret is Trueq
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()