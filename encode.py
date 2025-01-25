import os
import pickle
import face_recognition

# print(dir(face_recognition))

KNOWN_FACES_DIR = "known_faces"

known_encodings = []
known_usernames = []

for username in os.listdir(KNOWN_FACES_DIR):
    if not username[0] == '.':
        user_folder = os.path.join(KNOWN_FACES_DIR, username)
        if os.path.isdir(user_folder):
            for image in os.listdir(user_folder):
                if not image[0] == '.':
                    image_path = os.path.join(user_folder, image)
                    
                    load = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(load)
                    
                    known_encodings.append(encoding)
                    known_usernames.append(username)

data = {"encodings": known_encodings, "usernames": known_usernames}

with open("face_database.pkl", "wb") as file:
    pickle.dump(data, file)


