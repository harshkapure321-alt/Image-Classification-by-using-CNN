"""
CNN Training Script for Image Classification.
Loads dataset from folder structure, applies augmentation and preprocessing,
builds and trains a Convolutional Neural Network with callbacks, saves artifacts,
and plots training metrics.
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for server/headless execution
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN for Image Classification.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=150, help="Image height and width (square)")
    parser.add_argument("--model-dir", type=str, default="model", help="Directory to save trained model artifacts")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (0.0 to 1.0)")
    parser.add_argument("--model-type", type=str, default="mobilenet", choices=["custom", "mobilenet"], help="Model backbone: 'custom' CNN or 'mobilenet' transfer learning")
    return parser.parse_args()


def verify_dataset(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"\n[ERROR] Dataset directory '{dataset_dir}' does not exist.")
        print("Please arrange your dataset following this structure:")
        print("  dataset/")
        print("  ├── class_1/")
        print("  ├── class_2/")
        print("  └── class_3/")
        print("\nOr run: python generate_sample_dataset.py to generate sample data.")
        sys.exit(1)

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if len(subdirs) < 2:
        print(f"\n[ERROR] Dataset must contain at least 2 class subdirectories. Found: {subdirs}")
        print("Run: python generate_sample_dataset.py to create sample classes.")
        sys.exit(1)

    total_images = 0
    for d in subdirs:
        class_path = os.path.join(dataset_dir, d)
        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        images = [f for f in os.listdir(class_path) if os.path.splitext(f)[1] in valid_exts]
        print(f"Found {len(images)} images in class '{d}'")
        total_images += len(images)

    if total_images == 0:
        print(f"\n[ERROR] No valid images (.jpg, .jpeg, .png) found in {dataset_dir}.")
        sys.exit(1)

    return subdirs, total_images


def build_cnn_model(img_size, num_classes):
    """
    Constructs a Convolutional Neural Network with data augmentation,
    feature extraction blocks (Conv2D + MaxPool + BatchNorm), and a dense classification head.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    
    # In-graph Augmentation & Rescaling
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="dropout_conv")(x)

    # Dense Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Image_Classifier")
    return model


def build_mobilenet_transfer_model(img_size, num_classes):
    """
    Constructs a Convolutional Neural Network utilizing MobileNetV2 with pre-trained
    ImageNet weights and a fine-tuned dense classification head for rich real-world feature extraction.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = data_augmentation(inputs)
    # Scale inputs from [0, 255] to [-1, 1] as expected by MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Animal_Classifier")
    return model



def plot_and_save_history(history, output_path):
    """Plots and saves training and validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2, linestyle="--")
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ea580c", linewidth=2, linestyle="--")
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Training metrics plot saved to: {output_path}")


def main():
    args = parse_arguments()
    print("=" * 60)
    print("      CNN Image Classification - Model Training      ")
    print("=" * 60)

    # 1. Dataset Verification
    classes, total_images = verify_dataset(args.dataset)
    num_classes = len(classes)
    print(f"Total images: {total_images} across {num_classes} classes: {classes}")

    # Adjust batch size if dataset is smaller than default
    batch_size = min(args.batch_size, max(4, total_images // (num_classes * 2)))
    image_size = (args.img_size, args.img_size)

    print(f"\nLoading dataset with image size {image_size} and batch size {batch_size}...")

    # 2. Load Train and Validation Datasets
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        args.dataset,
        validation_split=args.val_split,
        subset="both",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    class_names = train_ds.class_names
    print(f"Detected class order: {class_names}")

    # Configure dataset performance (caching & prefetching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Create Model & Output Directories
    os.makedirs(args.model_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_dir, "cnn_model.keras")
    classes_save_path = os.path.join(args.model_dir, "class_names.json")
    plot_save_path = os.path.join(args.model_dir, "training_history.png")

    # Save class names JSON
    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"[OK] Class names saved to: {classes_save_path}")

    # 4. Build Model
    if args.model_type == "mobilenet":
        print("\n[Architecture] Using MobileNetV2 pre-trained CNN feature extractor (Transfer Learning)...")
        model = build_mobilenet_transfer_model(args.img_size, num_classes)
    else:
        print("\n[Architecture] Using Custom CNN Architecture...")
        model = build_cnn_model(args.img_size, num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # 5. Callbacks: EarlyStopping and ModelCheckpoint
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    # 6. Train Model
    print(f"\nStarting training for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Ensure best/final model is persisted
    model.save(model_save_path)
    print(f"[OK] Trained model successfully saved to: {model_save_path}")

    # 7. Evaluate and Plot
    print("\nEvaluating model on validation set...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    plot_and_save_history(history, plot_save_path)

    print("\n" + "=" * 60)
    print("Training Complete! Artifacts ready:")
    print(f"  - Model:         {model_save_path}")
    print(f"  - Class Names:   {classes_save_path}")
    print(f"  - History Chart: {plot_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
