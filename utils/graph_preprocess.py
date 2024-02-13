from tqdm import tqdm
import cv2
import numpy as np
import dlib

def detect_landmarks(images, predictor_path="shape_predictor_68_face_landmarks.dat"):
    landmarks = []
    for image in tqdm(images, desc="Detecting Landmarks"):
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = gray.astype(np.uint8)
      predictor = dlib.shape_predictor(predictor_path)
      height, width = gray.shape[:2]
      rect = dlib.rectangle(0, 0, width - 1, height - 1)
      landmark = predictor(gray, rect)
      landmark = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
      landmarks.append(landmark)
    return landmarks

def create_adjacency_matrix(landmarks):
  adjacency_matrices = []
  for landmark in tqdm(landmarks, desc="Creating Adjacency Matrix"):
    num_landmarks = len(landmark)
    adjacency_matrix = np.zeros((num_landmarks, num_landmarks))

    for i in range(num_landmarks):
      for j in range(i + 1, num_landmarks):
          distance = np.linalg.norm(np.array(landmark[i]) - np.array(landmark[j]))
          adjacency_matrix[i, j] = distance
          adjacency_matrix[j, i] = distance

    adjacency_matrices.append(adjacency_matrix)

  return adjacency_matrices