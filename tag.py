import pickle
import face_recognition
import numpy as np
import cv2

with open("face_database.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_usernames = data["usernames"]

def identify(image_path, known_encodings, known_usernames, tolerance=0.6):
    # load = face_recognition.load_image_file(image_path)

    load = cv2.imread(image_path)
    # load = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(load)
    face_encodings = face_recognition.face_encodings(load, face_locations)

    identified = []
    locations = []

    l = 0

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        best = None
        min_dist = float('inf')
        for i, (match, dist) in enumerate(zip(matches, face_distances)):
            if match and dist < min_dist:
                min_dist = dist
                best = i
        
        if best != None:
            username = known_usernames[best]
            identified.append(username)
        else:
            identified.append("Unknown")

        locations.append(face_locations[l])
        l += 1
    
    return (identified, locations)

def tag(image_path, identified, locations, output_path="tagged.jpg"):
    image = cv2.imread(image_path)

    for i in range(len(identified)):
        x = (locations[i][1] + locations[i][3]) // 2
        y = locations[i][2]

        (text_w, text_h), baseline = cv2.getTextSize(identified[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x -= text_w

        cv2.rectangle(image, (x, y + text_h), (int(x + 3.5 * text_w), y - 3 * text_h), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, identified[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, image)


a = []
for enc in known_encodings:
    a.append(np.array(enc).flatten())

new_image_path = "new4.jpeg"
results = identify(new_image_path, known_encodings=a, known_usernames=known_usernames, tolerance=0.6)
print(results)
tag(new_image_path, identified=results[0], locations=results[1])
