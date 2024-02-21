import time

import numpy as np
import cv2
import os
from pathlib import Path
import json
from plyfile import PlyData, PlyElement
from numba import njit
import _queue
import numba


def load_camera_matrix(path):
    # Opening JSON file and read the camera JSON file
    with open(path, 'r') as openfile:
        # Reading from json file
        camera_parameters = json.load(openfile)
        K = np.asarray([[camera_parameters["fx"], 0, camera_parameters["cx"]],
                        [0, camera_parameters["fy"], camera_parameters["cy"]],
                        [0, 0, 1]], np.float64)
    return K


def load_model_keypoints(path_to_model_keypoints):
    models_keypoints = {}
    for filename in os.listdir(path_to_model_keypoints):
        full_path = os.path.join(path_to_model_keypoints, filename)
        filename_without_extension = Path(filename).stem.replace("_kps", "")
        keypoints = []
        if os.path.isfile(full_path):
            with open(full_path, 'r') as openfile:
                for line in openfile:
                    x,y,z = line.split(" ")
                    keypoints.append([x,y,z])

            models_keypoints[filename_without_extension] = np.array(keypoints)

    return models_keypoints


def overlay_images(foreground_img, background_img, alpha=0.7, beta=1):

    foreground_img = foreground_img.astype(np.uint8)
    background_img = background_img.astype(np.uint8)

    result = cv2.addWeighted(foreground_img, alpha, background_img, beta, 0.0)
    result = result.astype(np.uint8)

    return result


def load_visualization_objects(models_path):
    assembly_models_face_vertex_indices = {}
    scaled_vertices = {}
    for filename in os.listdir(models_path):
        full_path = os.path.join(models_path, filename)
        filename_without_extension = Path(filename).stem.replace("_downsampled", "")
        # Load the PLY-Modell
        assembly_model = PlyData.read(full_path)

        # Define the data type so that numba can use it later on.
        np_vertex_indices = []
        vertex_indices = assembly_model['face']['vertex_indices']
        for index in vertex_indices:
            np_vertex_indices.append(np.array(index, np.float64))
        np_vertex_indices = np.array(np_vertex_indices)

        assembly_models_face_vertex_indices[filename_without_extension] = np_vertex_indices
        vertices = np.array(assembly_model['vertex'].data)

        # Scale the ply model
        scale_x = 0.001
        scale_y = 0.001
        scale_z = 0.001

        scaled_vertices[filename_without_extension] = []
        for vertex in vertices:
            scaled_vertex = np.array([vertex[0] * scale_x, vertex[1] * scale_y, vertex[2] * scale_z])
            scaled_vertices[filename_without_extension].append(scaled_vertex)

    return scaled_vertices, assembly_models_face_vertex_indices


@njit
def calculate_face_points(rotation_matrix, translation, intrinsic_matrix, scaled_vertices, assembly_model_face_vertex_indices):
    image_vertices = []
    for vertex in scaled_vertices:
        # Rotate and translate vertices
        vertex = np.dot(rotation_matrix, vertex)
        image_vertex = vertex + translation
        # Calculate position of 3D Modell points on image plane
        image_vertex = intrinsic_matrix @ image_vertex / image_vertex[2]
        image_vertices.append(image_vertex.astype(np.int32))

    point_list = []
    for face in assembly_model_face_vertex_indices:
        # Collect the points (u,v on the image) for each face of the model
        points = np.empty((0, 2), dtype=np.int32)
        for idx in face:
            points = np.append(points, image_vertices[int(idx)][:2].reshape(1,2), axis=0)
        point_list.append(points)

    return point_list


def display_3d_model_on_image(rotation_matrix, translation, intrinsic_matrix, scaled_vertices, assembly_model_face_vertex_indices, image, colour=[0, 0, 200]):
    # Call extra function to allow speed up with numba
    points_list = calculate_face_points(rotation_matrix, translation, intrinsic_matrix, numba.typed.List(scaled_vertices), assembly_model_face_vertex_indices)
    # Draw the filling between the calculated points (extra because of use of numba)
    for points in points_list:
        cv2.fillConvexPoly(image, points, colour)


def create_visualization(image, final_state_prediction, yolo_assembly_states, transformation_matrices, assembly_models_face_vertex_indices, intrinsic_matrix, scaled_vertices):
    blank_image = np.zeros((image.shape[0], image.shape[1], 3))

    cv2.putText(image, final_state_prediction, org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    current_assembly_states = yolo_assembly_states[-1]
    for state in current_assembly_states:
        # Display the state bounding boxes
        x1 = state["bbox_in_xyxy"][0].astype(np.int32)
        y1 = state["bbox_in_xyxy"][1].astype(np.int32)
        x2 = state["bbox_in_xyxy"][2].astype(np.int32)
        y2 = state["bbox_in_xyxy"][3].astype(np.int32)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(81, 35, 25), thickness=2)
        cv2.putText(image, state["state_name"], org=(x1, y1 - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(81, 35, 25), thickness=2, lineType=cv2.LINE_AA)

    start_time = time.time()

    for name in transformation_matrices:
        for entry in transformation_matrices[name]:
            transformation_matrix = entry["running_mean_transformation"]
            rotation_matrix = np.array(transformation_matrix[:, :3], np.float64)
            translation = np.array(transformation_matrix[:, 3], np.float64)
            # Display the 3D Models on the image. Does not directly draw on to the image because it should be transparent.
            display_3d_model_on_image(rotation_matrix, translation, intrinsic_matrix, scaled_vertices[name], assembly_models_face_vertex_indices[name], blank_image)

    print("Numba function time:  " + str(time.time() - start_time))

    image = overlay_images(blank_image, image)

    return image


def visualization_process(input_queue, output_queue, intrinsic_matrix, visualization_object_path, ):
    scaled_vertices, assembly_models_face_vertex_indices = load_visualization_objects(visualization_object_path)
    _intrinsic_matrix = intrinsic_matrix

    while True:
        try:
            received_data = input_queue.get(block=False, timeout=None)
        except _queue.Empty:
            received_data = None

        if received_data is not None:
            start_time = time.time()
            visualization_image = received_data["visualization_image"]
            yolo_assembly_states = received_data["yolo_assembly_states"]
            final_state_prediction = received_data["final_assembly_prediction"]
            transformation_matrices = received_data["transformation_matrices"]

            # Add borders of depth camera
            cv2.line(visualization_image, (60, 360), (300, 0), color=(0, 0, 255), thickness=3)
            cv2.line(visualization_image, (60, 360), (300, 720), color=(0, 0, 255), thickness=3)
            cv2.line(visualization_image, (1220, 360), (980, 0), color=(0, 0, 255), thickness=3)
            cv2.line(visualization_image, (1220, 360), (980, 720), color=(0, 0, 255), thickness=3)

            visualization_image = create_visualization(visualization_image, final_state_prediction, yolo_assembly_states, transformation_matrices, assembly_models_face_vertex_indices, _intrinsic_matrix, scaled_vertices)
            output_queue.put(visualization_image)
            print("Visualization process time: " + str(time.time() - start_time))
        else:
            time.sleep(0.005)
