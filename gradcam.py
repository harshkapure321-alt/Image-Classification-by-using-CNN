"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.
Generates visual explanations for CNN predictions by highlighting
the salient regions that contributed to the model's classification.
"""

import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def get_last_conv_layer_and_submodel(model):
    """
    Identifies the last convolutional layer in the model, handling both
    nested backbones (e.g. MobileNetV2) and custom flat CNN architectures.
    Returns (target_layer_name, sub_model, is_nested, backbone_name).
    """
    # Check for nested functional backbone (e.g., mobilenetv2)
    nested_backbone = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower() or (hasattr(layer, "layers") and len(layer.layers) > 10):
            nested_backbone = layer
            break

    if nested_backbone is not None:
        # Find the last Conv2D or ReLU activation inside the backbone
        for sub_layer in reversed(nested_backbone.layers):
            if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU)):
                return sub_layer.name, nested_backbone, True, nested_backbone.name

    # Flat custom CNN: search top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, model, False, None

    return None, None, False, None


def compute_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Computes a 2D Grad-CAM heatmap for a given input tensor.
    img_array: numpy array of shape (1, H, W, 3) in [0, 255] float32 range.
    pred_index: target class index (default: top predicted class).
    """
    target_layer_name, backbone, is_nested, _ = get_last_conv_layer_and_submodel(model)
    if target_layer_name is None:
        return None

    try:
        if is_nested:
            # Backbone is a sub-model (like MobileNetV2)
            target_conv_layer = backbone.get_layer(target_layer_name)
            sub_feature_model = tf.keras.Model(
                inputs=backbone.inputs,
                outputs=[target_conv_layer.output, backbone.output]
            )

            # Follow the forward graph of the full model
            with tf.GradientTape() as tape:
                # 1. Augmentation (if present)
                x = img_array
                if "data_augmentation" in [l.name for l in model.layers]:
                    x = model.get_layer("data_augmentation")(x, training=False)
                
                # 2. Rescaling / Preprocessing
                if "rescaling" in [l.name for l in model.layers]:
                    x = model.get_layer("rescaling")(x)
                else:
                    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

                # 3. Backbone forward pass
                conv_outputs, backbone_output = sub_feature_model(x)
                tape.watch(conv_outputs)

                # 4. Dense head pass
                pooled = model.get_layer("global_avg_pool")(backbone_output)
                if "head_dropout" in [l.name for l in model.layers]:
                    pooled = model.get_layer("head_dropout")(pooled, training=False)
                predictions = model.get_layer("predictions")(pooled)

                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            # Compute gradients of class score with respect to feature maps
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                return None

            # Channel-wise mean pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            # Weight the feature maps by their pooled gradients
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        else:
            # Flat Custom CNN architecture
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(target_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Apply ReLU to heatmap (focus only on features that positively contribute)
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    except Exception as e:
        print(f"[Grad-CAM Error] {e}")
        return None


def generate_gradcam_overlay_base64(model, img_array, original_pil_img, pred_index=None, alpha=0.45):
    """
    Computes the Grad-CAM heatmap, applies a jet colormap, overlays it
    on the original PIL image, and returns a base64 encoded PNG data URI.
    """
    heatmap = compute_gradcam_heatmap(model, img_array, pred_index)
    if heatmap is None:
        return None

    try:
        # Resize heatmap to match original image dimensions
        orig_w, orig_h = original_pil_img.size
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_resized = heatmap_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_arr = np.array(heatmap_resized) / 255.0

        # Apply colormap (jet: blue -> green -> yellow -> red)
        colormap = cm.get_cmap("jet")
        colored_heatmap = colormap(heatmap_arr)[:, :, :3]  # Discard alpha
        colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)
        colored_img = Image.fromarray(colored_heatmap_uint8)

        # Blend original RGB image with colored heatmap
        original_rgb = original_pil_img.convert("RGB")
        blended_img = Image.blend(original_rgb, colored_img, alpha=alpha)

        # Encode to Base64 PNG
        buffered = io.BytesIO()
        blended_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"[Grad-CAM Overlay Error] {e}")
        return None
