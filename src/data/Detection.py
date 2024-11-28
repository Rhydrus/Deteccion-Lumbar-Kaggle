import os
import pandas as pd
import numpy as np
import random
import torch
import glob
import pydicom
import albumentations as A
import cv2
import tqdm
from types import SimpleNamespace
from src.visualization.DetectionVisualize import visualize

from IPython.display import display

class ModelDataset:
    def __init__(self):
        self.processed_data = ""
        self.coords_pretrain = ""
        self.train_folder = ""

    def set_seed(self, seed=1234):
        """
        Establece la semilla para asegurar la reproducibilidad.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def load_data(self, processed_data, coords_pretrain, train_folder, plane, reverse_sort=False):
        """
        Carga los datos necesarios para el entrenamiento.
        """
        self.processed_data = processed_data
        self.train_folder = train_folder

        # Cargar metadatos
        df = pd.read_csv(coords_pretrain)
        df = df.sort_values(["source", "filename", "level"]).reset_index(drop=True)
        df["filename"] = df["filename"].str.replace(".jpg", ".npy", regex=False)
        df["series_id"] = df["source"] + "_" + df["filename"].str.split(".").str[0]
        self.coords_pretrain = df
        print("----- IMGS per source -----")
        display((df.source.value_counts() / 5).astype(int).reset_index())
        return df

    def model_config(self, frames, epochs, lr, batch_size, backbone, seed):
        """
        Configura los parámetros del modelo.
        """
        cfg = SimpleNamespace(
            images_dir=self.processed_data,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            n_frames=frames,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            backbone=backbone,
            seed=seed,
        )
        return cfg

    def model_split(self):
        """
        Divide los datos en conjuntos de entrenamiento y validación.
        """
        df = self.coords_pretrain
        train_df = df[df["source"] != "spider"]
        val_df = df[df["source"] == "spider"]

        return train_df, val_df

class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.records = self.load_coords(df)

    def __len__(self):
        return len(self.records)

    def load_coords(self, df):
        """
        Carga las coordenadas de los datos.
        """
        d = df.groupby("series_id")[["relative_x", "relative_y"]].apply(
            lambda x: list(x.itertuples(index=False, name=None))
        )
        records = {}
        for i, (k, v) in enumerate(d.items()):
            records[i] = {"series_id": k, "label": np.array(v).flatten()}
            assert len(v) == 5, f"Error: Expected 5 coordinates, got {len(v)} for {k}"
        return records

    def pad_image(self, img):
        """
        Rellena la imagen para que tenga el tamaño adecuado.
        """
        n = img.shape[-1]
        if n >= self.cfg.n_frames:
            start_idx = (n - self.cfg.n_frames) // 2
            return img[:, :, start_idx:start_idx + self.cfg.n_frames]
        else:
            pad_left = (self.cfg.n_frames - n) // 2
            pad_right = self.cfg.n_frames - n - pad_left
            return np.pad(
                img,
                ((0, 0), (0, 0), (pad_left, pad_right)),
                "constant",
                constant_values=0,
            )

    def load_img(self, source, series_id):
        """
        Carga una imagen desde un archivo .npy.
        """
        fname = os.path.join(
            self.cfg.images_dir, f"processed_{source}/{series_id}.npy"
        )
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Image file not found: {fname}")
        img = np.load(fname).astype(np.float32)
        img = self.pad_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        return img

    def __getitem__(self, idx):
        """
        Obtiene un elemento del dataset.
        """
        d = self.records[idx]
        label = d["label"]
        source = d["series_id"].split("_")[0]
        series_id = "_".join(d["series_id"].split("_")[1:])

        img = self.load_img(source, series_id)
        return {
            "img": img,
            "label": label,
        }

class ModelTrainer:
    def __init__(self, Model, Train_loader, Val_loader, Optimizer, Criterion, cfg, save_model=False):
        self.model = Model
        self.train_loader = Train_loader
        self.val_loader = Val_loader
        self.optimizer = Optimizer
        self.criterion = Criterion
        self.cfg = cfg
        self.save_model = save_model

    def batch_to_device(self, batch, device, skip_keys=[]):
        """
        Mueve un batch al dispositivo especificado.
        """
        batch_dict = {}
        for key in batch:
            batch_dict[key] = batch[key] if key in skip_keys else batch[key].to(device)
        return batch_dict

    def train_model(self):
        """
        Entrena el modelo.
        """
        for epoch in range(self.cfg.epochs + 1):
            # Bucle de entrenamiento
            loss = torch.tensor(0.0, device=self.cfg.device)
            if epoch != 0:
                train_model = self.model.train()
                for batch in tqdm.tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
                    batch = self.batch_to_device(batch, self.cfg.device)
                    self.optimizer.zero_grad()

                    x_out = train_model(batch["img"].float())
                    x_out = torch.sigmoid(x_out)

                    loss = self.criterion(x_out, batch["label"].float())
                    loss.backward()
                    self.optimizer.step()

            val_loss = 0
            # Bucle de validación
            with torch.no_grad():
                val_model = self.model.eval()
                for batch in tqdm.tqdm(self.val_loader, desc="Validating"):
                    batch = self.batch_to_device(batch, self.cfg.device)

                    pred = val_model(batch["img"].float())
                    pred = torch.sigmoid(pred)

                    val_loss += self.criterion(pred, batch["label"].float()).item()
                val_loss /= len(self.val_loader)

            visualize.visualize_training(self, self.cfg, batch, pred, epoch)
            
            print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
        print("Training complete...")

        if self.save_model:
            f = "{}_{}.pt".format(self.cfg.backbone, self.cfg.seed)
            torch.save(self.model.state_dict(), f)
            print("Saved weights: {}".format(f))

    def predict_model(self, model, weights, path, device):
        """
        Realiza predicciones utilizando un modelo entrenado para un estudio específico y visualiza las predicciones.
        
        :param model: Modelo a utilizar para predicción.
        :param weights: Ruta al archivo de pesos del modelo.
        :param path: Carpeta donde están las imágenes DICOM del estudio.
        :param device: Dispositivo a utilizar (CPU o GPU).
        """
        # Cargar los pesos del modelo
        state_dict = torch.load(weights, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()  # Poner el modelo en modo evaluación
        print(f"Pesos cargados desde: {weights}")

        # Cargar las imágenes DICOM del estudio
        dicom_files = sorted(glob.glob(os.path.join(path, "*.dcm")))
        if not dicom_files:
            raise FileNotFoundError(f"No se encontraron archivos .dcm en la carpeta: {path}")
        print(f"Se encontraron {len(dicom_files)} imágenes DICOM en {path}")

        predictions = []

        # Realizar predicciones y visualizarlas
        with torch.no_grad():
            for idx, file in enumerate(tqdm.tqdm(dicom_files, desc="Realizando predicciones")):
                # Leer archivo DICOM
                dicom = pydicom.dcmread(file)

                # Convertir el archivo DICOM a una imagen 2D en escala de grises
                img = dicom.pixel_array.astype(np.float32)

                # Normalizar la imagen
                img = cv2.resize(img, (224, 224))  # Redimensionar a las dimensiones esperadas
                img = img / np.max(img)  # Normalizar a rango [0, 1]

                # Convertir la imagen a formato [C, H, W]
                img = np.expand_dims(img, axis=0)  # Añadir dimensión de canal
                img = np.repeat(img, 3, axis=0)   # Repetir canal para simular RGB
                img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # Añadir dimensión batch

                # Obtener predicción del modelo
                output = model(img_tensor.float())
                output = torch.sigmoid(output).cpu().numpy()
                predictions.append(output)

                # Visualizar la predicción
                visualize.visualize_prediction(
                    img=img_tensor.squeeze(0).cpu().numpy(),  # Convertir tensor a numpy
                    pred=output.squeeze(),  # Coordenadas predichas
                    idx=idx
                )

        print(f"Predicciones completadas. Total de predicciones: {len(predictions)}")
        return np.vstack(predictions)
