import os
import tensorflow as tf


def deploy_model_on_rpi(model_path, output_path=None):
    """
    Deploy a TensorFlow model to Raspberry Pi Pico by converting it to TensorFlow Lite format.

    Args:
        model_path (str): Path to the TensorFlow model (.h5 file)
        output_path (str, optional): Path to save the quantized TFLite model.
                                    Defaults to '../model/model_quantized.tflite'.

    Returns:
        str: Path to the saved TFLite model

    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For any other errors during conversion or saving
    """
    if output_path is None:
        output_path = '../model/model_quantized.tflite'

    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load Keras model from .h5 file WITHOUT compiling (prevents loss function errors)
        model = tf.keras.models.load_model(model_path, compile=False)

        # Convert the model to TensorFlow Lite format with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply default quantization
        tflite_model = converter.convert()

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the quantized model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model successfully converted and saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during model deployment: {e}")
        raise
