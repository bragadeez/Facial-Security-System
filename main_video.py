import cv2
from simple_facerec import SimpleFacerec
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# loads the camera using cv2 module
cap = cv2.VideoCapture(0)

# initialize face_locations and face_names to empty lists
face_locations = []
face_names = []

# sending unknown faces to mail
sender_email = "e0122057@sret.edu.in"
sender_password = "jliq ysxx ixwd ogoa"
receiver_email = "bragu2004@gmail.com"

while True:
    ret, frame = cap.read()

    # checks if the frame is not empty
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        # detects faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        # check if any face is unknown
        is_unknown_detected = "Unknown" in face_names

        if is_unknown_detected:
            # capture and save the frame as "unknown_face.jpg"
            cv2.imwrite("unknown_face.jpg", frame)

            # send email when an unknown face is detected
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = "Unknown Face Detected"

            text = "Unknown face was detected. Here is an image of the unknown person:"
            message.attach(MIMEText(text, "plain"))

            # attach the captured image
            image_filename = "unknown_face.jpg"
            with open(image_filename, "rb") as image_file:
                image = MIMEImage(image_file.read(), name=os.path.basename(image_filename))
            message.attach(image)

            # create an SMTP client and send the email
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            server.quit()

    # continue with displaying known faces as before
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
