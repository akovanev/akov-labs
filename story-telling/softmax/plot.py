import matplotlib.pyplot as plt

def plot_image(X_test, index, true_label, pred_probs, predicted):
    # Display image + prediction
    img = X_test[index].reshape(28, 28)  # Use the same index here
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')

    probs_list = [f"{p:.4f}" for p in pred_probs[0].numpy()]
    probs_str = " ".join(probs_list[:10])
    plt.title(
        f"Index: {index}\n"
        f"True: {int(true_label)}\n"
        f"Predicted: {predicted.item()}\n"
        f"Probs: {probs_str}"
    )
    plt.axis('off')
    plt.show()