import matplotlib.pyplot as plt
import os

class visualize():
    def __init__(self):
        pass
    
    def visualize_training(self,cfg,batch, pred, epoch):

        mid = cfg.n_frames // 2

        # Plot
        for idx in range(1):

            # Select Data
            img = batch["img"][idx, mid, :, :].cpu().numpy() * 255
            cs_true = batch["label"][idx, ...].cpu().numpy() * 256
            cs = pred[idx, ...].cpu().numpy() * 256

            coords_list = [("TRUE", "lightblue", cs_true), ("PRED", "orange", cs)]
            text_labels = [str(x) for x in range(1, 6)]

            # Plot coords
            fig, axes = plt.subplots(1, len(coords_list), figsize=(10, 4))
            fig.suptitle("EPOCH: {}".format(epoch))
            for ax, (title, color, coords) in zip(axes, coords_list):
                ax.imshow(img, cmap="gray")
                ax.scatter(coords[0::2], coords[1::2], c=color, s=50)
                ax.axis("off")
                ax.set_title(title)

                # Add text labels near the coordinates
                for i, (x, y) in enumerate(zip(coords[0::2], coords[1::2])):
                    if i < len(text_labels):  # Ensure there are enough labels
                        ax.text(
                            x + 10,
                            y,
                            text_labels[i],
                            color="white",
                            fontsize=15,
                            bbox=dict(facecolor="black", alpha=0.5),
                        )
            
            fig.suptitle("EPOCH: {}".format(epoch))
            save_path = os.path.join(r'C:\Users\Carlo\Desktop\Proyectos\Lumbar\reports\Figures\train_images', f"epoch_{epoch}_batch_{idx}.png")
            plt.savefig(save_path)
            plt.show()
            

        return
        
    @staticmethod
    def visualize_prediction(img, pred, idx):
        """
        Visualiza una predicción con etiquetas sobre la imagen.

        :param img: Imagen en formato numpy.
        :param pred: Coordenadas predichas por el modelo.
        :param idx: Índice de la imagen para el título.
        """
        img = img.transpose(1, 2, 0)[:, :, 0] * 255  # Convertir a escala de grises
        coords = pred * img.shape[0]  # Ajustar las coordenadas al tamaño de la imagen

        # Configuración de etiquetas
        text_labels = ["L1", "L2", "L3", "L4", "L5"]  # Ejemplo para las vértebras lumbares

        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        # Dibujar coordenadas y etiquetas
        for i, (x, y) in enumerate(zip(coords[0::2], coords[1::2])):
            plt.scatter(x, y, c="orange", s=50)
            if i < len(text_labels):
                plt.text(
                    x + 10, y, text_labels[i], color="white", fontsize=15,
                    bbox=dict(facecolor="black", alpha=0.5)
                )

        plt.title(f"Predicción - Imagen {idx + 1}")
        plt.show()
        
    @staticmethod
    def plot_accuracies(train_accuracies, val_accuracies):
        """
        Grafica el accuracy de entrenamiento y validación por epoch.
        
        :param train_accuracies: Lista de accuracies en el conjunto de entrenamiento.
        :param val_accuracies: Lista de accuracies en el conjunto de validación.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(train_accuracies, label="Entrenamiento", marker="o")
        plt.plot(val_accuracies, label="Validación", marker="o")
        plt.title("Precisión durante el Entrenamiento")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()
