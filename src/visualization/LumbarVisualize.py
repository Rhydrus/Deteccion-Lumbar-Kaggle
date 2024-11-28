import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import os
from tqdm import tqdm  # Asegúrate de importar tqdm correctamente
import json
import glob
import pydicom

import pandas as pd
import random
import numpy as np
import torch
import albumentations as A
import math

class ConfigVisualize:
    """
    Clase para configurar y manejar la visualización de datos de imágenes lumbares.
    """
    def __init__(self) -> None:
        """
        Inicializa la configuración de visualización cargando las descripciones de las series de entrenamiento,
        las carpetas de imágenes de entrenamiento y los datos de pacientes desde un archivo JSON.
        """
        # Actualiza las rutas según sea necesario
        self.train_series_description = pd.read_csv(r'C:\Users\Carlo\Desktop\Proyectos\Lumbar\data\csv\train_series_descriptions.csv')
        self.train_folders = os.listdir(r'C:\Users\Carlo\Desktop\Proyectos\Lumbar\data\raw_data\train_images')
        
        with open(r'C:\Users\Carlo\Desktop\Proyectos\Lumbar\data\filters\datos.json', 'r') as archivo:
            self.PatientsJSON = json.load(archivo)

    def get_patient_folder(self, patient_object):
        """
        Obtiene las imágenes DICOM y sus descripciones para un paciente específico.

        Args:
            patient_object (str): El identificador del paciente en el archivo JSON.

        Returns:
            dict: Un diccionario con las imágenes DICOM y sus descripciones.
        """
        patient_object = self.PatientsJSON[patient_object]
        
        im_list_dcm = {}
        for idx, i in enumerate(patient_object['SeriesInstanceUIDs']):
            im_list_dcm[i] = {'images': [], 'description': patient_object['SeriesDescriptions'][idx]}
            
            # Usar os.path.join para construir la ruta correctamente
            folder_path = os.path.join(patient_object['folder_path'], patient_object['SeriesInstanceUIDs'][idx])
            images = glob.glob(os.path.join(folder_path, '*.dcm'))
            
            for j in sorted(images, key=lambda x: int(os.path.basename(x).replace('.dcm', ''))):
                im_list_dcm[i]['images'].append({
                    'SOPInstanceUID': os.path.basename(j).replace('.dcm', ''), 
                    'dicom': pydicom.dcmread(j)
                })
        
        return im_list_dcm
    
class Visualize:
    """
    Clase para visualizar las imágenes DICOM de un paciente.
    """
    def __init__(self, patiend_folder) -> None:
        """
        Inicializa la visualización de las imágenes DICOM de un paciente.

        Args:
            patiend_folder (str): El identificador del paciente en el archivo JSON.
        """
        config_visualize = ConfigVisualize()
        self.patient_object = config_visualize.get_patient_folder(patiend_folder)
    
        # Este código se encarga de visualizar las coordenadas de las patologías en las imágenes
        for i in self.patient_object:
            self.display_images([x['dicom'].pixel_array for x in self.patient_object[i]['images']], self.patient_object[i]['description'])
    
    def display_images(self, images, title, max_images_per_row=4):
        """
        Muestra las imágenes DICOM en una cuadrícula.

        Args:
            images (list): Lista de matrices de píxeles de las imágenes DICOM.
            title (str): Título de la visualización.
            max_images_per_row (int): Número máximo de imágenes por fila.
        """
        if not images:
            print("No images to display.")
            return
        
        num_images = len(images)
        num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # División de techo

        fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))
        
        # Aplana la matriz de ejes para facilitar el bucle
        axes = axes.flatten() if num_rows > 1 else [axes]

        for idx, image in enumerate(images):
            ax = axes[idx]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

class coordinates_pathologies:

    """
    Clase para manejar y visualizar las coordenadas de las patologías en las imágenes DICOM.
    """
    def __init__(self, patient_folder) -> None:
        """
        Inicializa la clase y carga los datos del paciente.

        Args:
            patiend_folder (str): El identificador del paciente en el archivo JSON.
        """
        self.train_csv = pd.read_csv(r'C:\Users\Carlo\Desktop\Proyectos\Lumbar\data\csv\train.csv')
        self.train_coordinates = pd.read_csv(r"C:\Users\Carlo\Desktop\Proyectos\Lumbar\data\csv\train_label_coordinates.csv")
        
        config_visualize = ConfigVisualize()
        self.patient_id = int(patient_folder)
        self.patient_object = config_visualize.get_patient_folder(patient_folder)
        self.config_display_coor()

    def display_coor_on_img(self, c, i, title):
        """
        Muestra las coordenadas de las patologías en la imagen DICOM.

        Args:
            c (Series): Coordenadas de la patología.
            i (dict): Información de la imagen DICOM.
            title (str): Título de la visualización.
        """
        IMG = i['dicom'].pixel_array
        center_coordinates = (int(c['x']), int(c['y']))
        radius = 5
        color = (255, 0, 0)  # Red color in BGR
        thickness = 2

        # Normalize the image to 8-bit
        IMG_normalized = cv2.normalize(IMG, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        IMG_with_circle = cv2.circle(IMG_normalized.copy(), center_coordinates, radius, color, thickness)
        
        # Convert the image from BGR to RGB for correct color display in matplotlib
        IMG_with_circle = cv2.cvtColor(IMG_with_circle, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.imshow(IMG_with_circle)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.title(title)
        plt.show()
        
    def config_display_coor(self):
        """
        Configura y muestra las coordenadas de las patologías en las imágenes DICOM.
        """
        
        # Este código se encarga de visualizar las coordenadas de las patologías en las imágenes
        patient = self.train_csv[self.train_csv['study_id'] == int(self.patient_id)].iloc[0]
        coor_entries = self.train_coordinates[self.train_coordinates['study_id'] == int(patient['study_id'])]
        
        # Este código se encarga de visualizar las coordenadas de las patologías en las imágenes
        print("Only showing severe cases for this patient")
        print("Available keys in patient_object:", self.patient_object.keys())  # Imprime las claves disponibles
        for idc, c in coor_entries.iterrows():
            series_id = str(c['series_id'])
            if series_id in self.patient_object:
                for i in self.patient_object[series_id]['images']:
                    if int(i['SOPInstanceUID']) == int(c['instance_number']):
                        try:
                            patient_severity = patient[
                                f"{c['condition'].lower().replace(' ', '_')}_{c['level'].lower().replace('/', '_')}"
                            ]
                        except Exception as e:
                            patient_severity = "unknown severity"
                        title = f"{i['SOPInstanceUID']} \n{c['level']}, {c['condition']}: {patient_severity} \n{c['x']}, {c['y']}"
                        if patient_severity == 'Severe':
                            self.display_coor_on_img(c, i, title)
            else:
                print(f"Series ID {series_id} not found in patient_object.")

class lumbar_coordinates:
    
    def __init__(self, train_folder,train_descriptions,coords) -> None:
        
        self.train_folder = train_folder
        self.train_descriptions = train_descriptions
        self.coords = coords
        
        self.resize_transform = A.Compose([
            A.LongestMaxSize(max_size=256, interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.PadIfNeeded( min_height=256,min_width=256,border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True,),
            ])

    def set_seed(self, seed = 1234):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
    def convert_to_8bit(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return (x * 255).astype("uint8")
    
    def load_dicom_stack(self, dicom_folder, plane, reverse_sort=False):
        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
        positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
        # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
        # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
        idx = np.argsort(-positions if reverse_sort else positions)
        ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
        array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
        array = array[idx]
        return {
            "array": self.convert_to_8bit(array),
            "positions": ipp,
            "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float"),
        }
    
    def angle_of_line(self, x1, y1, x2, y2):
        return math.degrees(math.atan2(-(y2 - y1), x2 - x1))
    
    def plot_img(self, img, coords_temp):
        # Plot img
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        h, w = img.shape

        # Kepoints as pairs
        p = (
            coords_temp.groupby("level")
            .apply(lambda g: list(zip(g["relative_x"], g["relative_y"])))
            .reset_index(drop=False, name="vals")
        )

        # Plot keypoints
        for _, row in p.iterrows():
            level = row["level"]
            x = [_[0] * w for _ in row["vals"]]
            y = [_[1] * h for _ in row["vals"]]
            ax.plot(x, y, marker="o")
        ax.axis("off")
        plt.show()
        
    def crop_between_keypoints(self, img, keypoint1, keypoint2):
        h, w = img.shape
        x1, y1 = int(keypoint1[0]), int(keypoint1[1])
        x2, y2 = int(keypoint2[0]), int(keypoint2[1])

        # Calculate bounding box around the keypoints
        left = int(min(x1, x2))
        right = int(max(x1, x2))
        top = int(min(y1, y2) - (h * 0.1))
        bottom = int(max(y1, y2) + (h * 0.1))

        # Crop the image
        return img[top:bottom, left:right]
        
    def plot_5_crops(self, img, coords_temp):
        # Create a figure and axis for the grid
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1] * 5)

        # Plot the crops
        p = (
            coords_temp.groupby("level")
            .apply(lambda g: list(zip(g["relative_x"], g["relative_y"])))
            .reset_index(drop=False, name="vals")
        )
        for idx, (_, row) in enumerate(p.iterrows()):
            # Copy of img
            img_copy = img.copy()
            h, w = img.shape

            # Extract Keypoints
            level = row["level"]
            vals = sorted(row["vals"], key=lambda x: x[0])
            a, b = vals
            a = (a[0] * w, a[1] * h)
            b = (b[0] * w, b[1] * h)

            # Rotate
            rotate_angle = self.angle_of_line(a[0], a[1], b[0], b[1])
            transform = A.Compose(
                [
                    A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

            t = transform(image=img_copy, keypoints=[a, b])
            img_copy = t["image"]
            a, b = t["keypoints"]

            # Crop + Resize
            img_copy = self.crop_between_keypoints(img_copy, a, b)
            img_copy = self.resize_transform(image=img_copy)["image"]

            # Plot
            ax = plt.subplot(gs[idx])
            ax.imshow(img_copy, cmap="gray")
            ax.set_title(level)
            ax.axis("off")
        plt.show()

    def coordinate_visualise(self,dicom_folder, plane, reverse_sort = False):
        
        dfd = pd.read_csv(filepath_or_buffer=fr"{self.train_descriptions}")
        dfd = dfd[dfd.series_description == "Sagittal T2/STIR"]
        self.dfd = dfd.sample(frac=1, random_state=10).head(2)
        
        coords = pd.read_csv(fr"{self.coords}")
        coords = coords.sort_values(["series_id", "level", "side"]).reset_index(drop=True)
        self.coords = coords[["series_id", "level", "side", "relative_x", "relative_y"]]
        
        # Plot samples
        for idx, row in self.dfd.iterrows():
            try:
                print(
                    "-" * 25,
                    " STUDY_ID: {}, SERIES_ID: {} ".format(row.study_id, row.series_id),
                    "-" * 25,
                )
                sag_t2 = self.load_dicom_stack(
                    os.path.join(self.train_folder, str(object=row.study_id), str(row.series_id)),
                    plane="axial",
                )

                # Img + Coords
                img = sag_t2["array"][len(sag_t2["array"]) // 2]
                coords_temp = self.coords[self.coords["series_id"] == row.series_id].copy()

                # Plot
                self.plot_img(img, coords_temp)
                self.plot_5_crops(img, coords_temp)

            except Exception as e:
                print(e)
                pass