import face_recognition
import cv2
import glob
import re
from pytz import timezone
from datetime import datetime, date

video_capture = cv2.VideoCapture("rtsp://USERNAME:PASSWORD@192.168.1.64:554")
images = glob.glob('*.png')
known_face_encodings = []
known_face_names = []
for image in images:
  obama_image = face_recognition.load_image_file(image)
  obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
  known_face_encodings.append(obama_face_encoding)
  m = re.match(r"(?P<first_name>\w+).*", image)
  known_face_names.append(m.group('first_name'))
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
prevName = "Unknown"
while True:
  ret, frame = video_capture.read()
  small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
  rgb_small_frame = small_frame[:, :, ::-1]
  if process_this_frame:
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
      name = "Unknown"
      if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
      if prevName != name:
        print(str(datetime.now()))
        print(name)
      face_names.append(name)
      prevName = name
  process_this_frame = not process_this_frame
  for (top, right, bottom, left), name in zip(face_locations, face_names):
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
  cv2.imshow('Video', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
video_capture.release()
cv2.destroyAllWindows()
