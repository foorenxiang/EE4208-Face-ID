    import cv2

    import matplotlib.pyplot as plt
    import dlib
    from imutils import face_utils

    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    from joblib import load

    dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

    cascPath = dataLoc + "/haarcascade_frontalface_default.xml"
    eyePath = dataLoc + "/haarcascade_eye.xml"
    smilePath = dataLoc + "/haarcascade_smile.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)
    eyeCascade = cv2.CascadeClassifier(eyePath)
    smileCascade = cv2.CascadeClassifier(smilePath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            frame = cv2.flip(frame, 1)
            cv2.putText(frame,'Face',(1280-w-x, y), font, 2,(255,0,0),5)

            # smile = smileCascade.detectMultiScale(
            #     roi_gray,
            #     scaleFactor= 1.16,
            #     minNeighbors=35,
            #     minSize=(25, 25),
            #     flags=cv2.CASCADE_SCALE_IMAGE
            # )

            # for (sx, sy, sw, sh) in smile:
            #     cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            #     cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

            # eyes = eyeCascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #     cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
            cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
        # Display the resulting frame
        cv2.imshow('Video', cv2.flip(frame, 1))

        #######face identification#######
        faceVectors = list() #list containing faceVectors
        
        if len(faces):
            for (x, y, w, h) in faces:
                faceVector = list()  #vector of pixels for each face
                #convert face pixels to face vector
                for col in range(y,(y+h)):
                    for row in range(x,(x+w)):
                        faceVector.append(row)
                faceVectors.append(faceVector)


        print("Number of faces detected:")
        print(len(faceVectors))

        if len(faceVectors):
            print("\n Dimensions of each face vector:")
            print(list(map(lambda x : len(x), faceVectors)))


        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()