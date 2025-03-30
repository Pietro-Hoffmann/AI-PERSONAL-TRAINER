import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np # biblioteca para fazer manipulações de arrays (sera utilizado para manipular a imagens pois ela são lidas como array de matrizes, tendo a largura de pixels, comprimento de pixels e esquema de cores, no caso rgb então será 3 (Exemplo: 1200, 640, 3))
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

file_name='Teste_Supino.mp4'
model_path = 'pose_landmarker_full.task'




def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image





cap = cv2.VideoCapture(file_name)  #Realiza a leitura do arquivo

options = python.vision.PoseLandmarkerOptions(
   base_options=python.BaseOptions(model_asset_path=model_path),
   running_mode = python.vision.RunningMode.VIDEO
)

while (cap.isOpened()): # = enquanto o vídeo estiver aberto retorna True ou False
    ret, frame = cap.read() # Faz com que toda fez que o cap.read for rodado ele lê um frame diferente do vídeo
    if ret == True:
        cv2.imshow('Frame', frame) #vai mostrar essa imagem

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'): # esse comando vem do OpenCV
            break
    else: # caso o ret retornar falso, da um break no while
        break

cap.release() # Libera o vídeo
cv2.destroyAllWindows() # Fecha as janelas que foram abertas