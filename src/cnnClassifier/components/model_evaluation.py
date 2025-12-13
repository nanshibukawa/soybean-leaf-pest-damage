import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score
)
import tensorflow as tf

from cnnClassifier.entity.config_entity import ImageConfig, ModelConfig
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)


class ModelEvaluator:
    """
    Componente que realiza a avaliação do modelo treinado

    Args:
        model_config: Configuração do modelo
        image_config: Configuração das imagens
        class_names: Lista de nomes das classes para relatórios e visualizações

    """

    def __init__(
        self,
        model_config: ModelConfig,
        image_config: ImageConfig,
        class_names: List[str],
    ):
        self.model_config = model_config
        self.image_config = image_config
        self.class_names = class_names

    def evaluate(
        self,
        model: tf.keras.Model,
        validation_data: tf.data.Dataset,
        save_dir: Path,
        history: tf.keras.callbacks.History,
    ):
        """
        Executa a avaliação do modelo treinado

        Args:
            model: Modelo treinado a ser avaliado
            validation_data: Dados de validação para avaliação
            save_dir (Path): Diretório onde salvar os resultados

        """

        logger.info("🎯 Iniciando avaliação do modelo...")

        y_true, y_pred = self._predict(model, validation_data)

        metrics = self._calculate_metrics(y_true, y_pred)

        self._plot_all_results(y_true, y_pred, metrics, save_dir, history)

        report_path = save_dir / "evaluation_report.json"

        self._save_report(metrics, report_path)

        logger.info(f"✅ Avaliação concluída! Accuracy: {metrics['accuracy']:.4f}")

        return {
            "success": True,
            "metrics": metrics,
            "predictions": {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
        }

    def _predict(self, model, validation_data):
        """Faz predições no dataset de validação"""
        logger.info("🔍 Fazendo predições...")

        y_true, y_pred = [], []

        for batch_images, batch_labels in validation_data:
            predictions = model.predict(batch_images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted_classes)

        return np.array(y_true), np.array(y_pred)

    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas detalhadas"""
        logger.info("📊 Calculando métricas...")

        # Métricas essenciais
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # Relatório detalhado
        classification_rep = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_rep,
            "num_classes": len(np.unique(y_true)),
            "total_samples": len(y_true),
        }

    def _plot_all_results(self, y_true, y_pred, metrics, save_dir, history=None):
        """Gera TODAS as visualizações (métricas + training history)"""
        logger.info("📊 Gerando visualizações...")

        # PLOT 1: Métricas de Avaliação (Confusion Matrix + F1-Score)
        self._plot_evaluation_metrics(metrics, save_dir)

        # PLOT 2: Training History
        if history is not None:
            self._plot_training_history(history, save_dir)

    def _plot_evaluation_metrics(self, metrics, save_dir):
        """Plot das métricas de avaliação (confusion matrix + F1-Score)"""
        # TODO verify fig used
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Matriz de confusão
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax1,
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        ax1.set_title("Matriz de Confusão")
        ax1.set_xlabel("Predição")
        ax1.set_ylabel("Real")

        # Métricas por classe
        report = metrics["classification_report"]
        classes = self.class_names
        f1_scores = [report[cls]["f1-score"] for cls in classes]

        ax2.bar(classes, f1_scores, alpha=0.8, color=["steelblue", "orange", "green"])
        ax2.set_title("F1-Score por Classe")
        ax2.set_ylabel("F1-Score")
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_ylim(0, 1.0)

        # Adicionar valores nas barras
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(save_dir / "evaluation_metrics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_training_history(self, history, save_dir):
        """Plot do histórico de treinamento (Loss + Accuracy)"""
        logger.info("📈 Gerando gráficos de treinamento...")

        # Extrair dados do history
        acc = history.history.get("accuracy", [])
        val_acc = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])

        epochs = range(1, len(acc) + 1)

        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Accuracy
        ax1.plot(epochs, acc, "b-", label="Acurácia do Treino", linewidth=2)
        ax1.plot(epochs, val_acc, "r-", label="Acurácia da Validação", linewidth=2)
        ax1.set_title("Acurácia do Modelo por Época")
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("Acurácia")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Plot 2: Loss
        ax2.plot(epochs, loss, "b-", label="Loss do Treino", linewidth=2)
        ax2.plot(epochs, val_loss, "r-", label="Loss da Validação", linewidth=2)
        ax2.set_title("Loss do Modelo por Época")
        ax2.set_xlabel("Épocas")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "training_history.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"📈 Gráficos de treinamento salvos em: {save_dir}/training_history.png"
        )

    def _save_report(self, metrics, save_path):
        """Salva relatório JSON"""
        logger.info("💾 Salvando relatório...")

        # Relatório essencial
        report = {
            "model_info": {
                "model_name": self.model_config.model_name,
                "total_samples": metrics["total_samples"],
            },
            "overall_metrics": {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            },
            "per_class_metrics": {
                "classes": self.class_names,
                "precision": [
                    metrics["classification_report"][cls]["precision"]
                    for cls in self.class_names
                ],
                "recall": [
                    metrics["classification_report"][cls]["recall"]
                    for cls in self.class_names
                ],
                "f1_score": [
                    metrics["classification_report"][cls]["f1-score"]
                    for cls in self.class_names
                ],
                "support": [
                    metrics["classification_report"][cls]["support"]
                    for cls in self.class_names
                ],
            },
            "detailed_classification_report": metrics["classification_report"],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        logger.info(f"📄 Relatório salvo em: {save_path}")
        return report
