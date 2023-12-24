# pip install cmake
# pip install face-recognition
# pip install opencv-python
import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime
video_capture = cv2.VideoCapture(0)
# load known faces
subh_image = face_recognition.load_image_file("faces/subh.jpg")
subh_encoding = face_recognition.face_encodings(subh_image)[0]
naitik_image = face_recognition.load_image_file("faces/naitik.jpg")
naitik_encoding = face_recognition.face_encodings(naitik_image)[0]
himmo_image = face_recognition.load_image_file("faces/himmo.jpg")
himmo_encoding = face_recognition.face_encodings(himmo_image)[0]
anand_image = face_recognition.load_image_file("faces/anand.jpg")
anand_encoding = face_recognition.face_encodings(anand_image)[0]
nitin_image = face_recognition.load_image_file("faces/nitin.jpg")
nitin_encoding = face_recognition.face_encodings(nitin_image)[0]
ganesh_image = face_recognition.load_image_file("faces/ganesh.jpg")
ganesh_encoding = face_recognition.face_encodings(ganesh_image)[0]
taps_image = face_recognition.load_image_file("faces/taps.jpg")
taps_encoding = face_recognition.face_encodings(taps_image)[0]
suyash_image = face_recognition.load_image_file("faces/suyash.jpg")
suyash_encoding = face_recognition.face_encodings(suyash_image)[0]


known_faces_encoding = [subh_encoding,naitik_encoding,himmo_encoding,ganesh_encoding,nitin_encoding,anand_encoding,taps_encoding,suyash_encoding]
known_faces_name = ["Subhanshu","Naitik","Himalika","Ganesh","Nitin","Anand","TAPS","Suyash"]

# list of expected candidate
can = known_faces_name.copy()
face_location = []
face_encoding = []

# get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")


f = open(f"{current_date}.csv","a+",newline="")
lnwrite = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx = 0.20,fy = 0.20)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    # RECOGONIZE FACES
    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_location)

    for face_encoding in face_encodings:
        comp  = face_recognition.compare_faces(known_faces_encoding,face_encoding)
        face_dist = face_recognition.face_distance(known_faces_encoding,face_encoding)
        best_match_index = np.argmin(face_dist)

        if(comp[best_match_index]):
            name = known_faces_name[best_match_index]

        # add text if person is present
        if name in known_faces_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomleftcorneroftxt = (10,100)
            fontscale = 1.5
            fontcolor = (255,0,0)
            thikness = 2
            linetype = 2
            cv2.putText(frame , name + "Present",bottomleftcorneroftxt,font,fontscale,fontcolor,thikness,linetype)

            if name in can:
                can.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwrite.writerow([name , current_time])

    cv2.imshow("Present",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
