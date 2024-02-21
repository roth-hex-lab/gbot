import time
import cv2
import keyboard
import numpy as np
import json
from ultralytics import YOLO
import open3d as o3d
from numba import njit

from plyfile import PlyData, PlyElement

from Evaluation.pnp import *

from visulisation_helper_functions import *


def load_visualization_objects(models_path):
    assembly_models = {}
    scaled_vertices = {}
    for filename in os.listdir(models_path):
        full_path = os.path.join(models_path, filename)
        filename_without_extension = Path(filename).stem.replace("_downsampled", "")
        # Lade das PLY-Modell
        assembly_models[filename_without_extension] = PlyData.read(full_path)
        #assembly_model = PlyData.read("D:/students/2023-corell-niklas-mt_new/Synthetic_data_generation/resources/3Dprint/NanoViseV2/models/NanoViseV2_CLAMP_BASE.ply")
        vertices = np.array(assembly_models[filename_without_extension]['vertex'].data)

        # Skalierungsfaktoren für das 3D-Modell
        scale_x = 0.001  # Ändere die Skalierungsfaktoren entsprechend
        scale_y = 0.001
        scale_z = 0.001

        scaled_vertices[filename_without_extension] = []
        for vertex in vertices:
            scaled_vertex = np.array([vertex[0] * scale_x, vertex[1] * scale_y, vertex[2] * scale_z])
            scaled_vertices[filename_without_extension].append(scaled_vertex)

    return scaled_vertices, assembly_models


def display_3d_model_on_image(rotation_matrix, translation, intrinsic_matrix, image, scaled_vertices, assembly_model):
    image_vertices = []
    for vertex in scaled_vertices:
        vertex = np.dot(rotation_matrix, vertex)
        image_vertex = vertex + translation
        image_vertex = intrinsic_matrix @ image_vertex / image_vertex[2]
        image_vertices.append(image_vertex.astype(int))

    # Rendere das 3D-Modell als Oberfläche auf das Bild
    for face in assembly_model['face']['vertex_indices']:
        points = np.array([image_vertices[idx] for idx in face])
        points = np.array(points[:, :2], np.int32)
        cv2.fillConvexPoly(image, points, [0, 0, 200])


def main():
    model = YOLO("C:/Users/Student/Downloads/best.pt", task="pose")  # Continue Training, set resume in train method to True

    cam_port = 1
    image_width = 1280
    image_height = 720

    cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    models_keypoints = load_model_keypoints("../../Synthetic_data_generation/resources/3Dprint/NanoViseV2/Keypoints")

    intrinsic_matrix = load_camera_matrix("../../Synthetic_data_generation/our_camera.json")

    scaled_vertices, assembly_models = load_visualization_objects("../extrem_downsampled_models_for_visualization")

    while not keyboard.is_pressed("esc"):

        # reading the input using the camera
        cam_read_success, image = cam.read()

        # If image will detected without any error show result
        if cam_read_success:

            results = model(image, imgsz=(1280, 1280), stream=False, conf=0.5, show=False, classes=None)  # [9, 10, 11, 12, 13, 14, 15])  # predict on an image

            names = results[0].names
            result_keypoints = results[0].keypoints.xy.cpu().numpy()
            result_classes = results[0].boxes.cls.cpu().numpy()

            start_time = time.time()
            blank_image = np.zeros((image_height, image_width, 3))
            for idx, result_class in enumerate(result_classes):
                name = names[result_class]
                current_part_keypoints = result_keypoints[idx]

                # Draw keypoints
                #for keypoint in part_keypoints:
                #    image = cv2.circle(image, center=(int(keypoint[0]), int(keypoint[1])), radius=2, color=(0, 0, 255), thickness=-1)

                if name in models_keypoints.keys() and name in assembly_models.keys():
                    transformation_matrix = ransacpnp(models_keypoints[name], current_part_keypoints, intrinsic_matrix)

                    rotation_matrix = transformation_matrix[:,:3]
                    translation = transformation_matrix[:,3]

                    display_3d_model_on_image(rotation_matrix, translation, intrinsic_matrix, blank_image, scaled_vertices[name], assembly_models[name])

            image = overlay_images(blank_image, image)

            end_time = time.time()
            print(end_time - start_time)
            # showing result
            cv2.imshow("PosePredictor", image)
            cv2.waitKey(1)

        # If captured image is corrupted, moving to else part
        else:
            print("Image capturing failed.")

    # Destroying All the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
