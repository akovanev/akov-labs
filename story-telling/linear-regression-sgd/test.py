import torch
import numpy as np

def interpret_results(model, X_data, X_normalized_tensor, prices, noise_std):
    # Evaluate Model Performance on Raw Data
    model.eval()

    # 1. Extract Normalized Parameters
    with torch.no_grad():
        w_norm = model.weight.squeeze().cpu().numpy()
        b_norm = model.bias.item()

    # 2. Statistics for Un-normalization
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)

    # 3. The Recovery Math
    beta_recovered = w_norm / X_std
    intercept_recovered = b_norm - np.sum((w_norm * X_mean) / X_std)

    # 4. Predictions and Error Metrics
    with torch.no_grad():
        # Ensuring we compare Raw vs Raw
        y_pred_raw = model(X_normalized_tensor).numpy().flatten()
        y_true_raw = prices.flatten()

    mse_orig = np.mean((y_pred_raw - y_true_raw)**2)
    rmse_orig = np.sqrt(mse_orig)

    # --- INTERPRETATION PRINTING ---
    print("\n" + "="*30)
    print("   MODEL RECOVERY REPORT")
    print("="*30)

    print(f"{'Parameter':<15} | {'True':<10} | {'Recovered':<10} | {'Error %':<10}")
    print("-" * 55)

    # Assuming True betas = [0.85, 15] and Intercept = 10
    true_betas = [0.85, 15]
    true_intercept = 10

    for i, (t, r) in enumerate(zip(true_betas, beta_recovered)):
        err = abs(t - r) / t * 100
        print(f"Beta {i+1:<10} | {t:<10.2f} | {r:<10.4f} | {err:<10.2f}%")

    int_err = abs(true_intercept - intercept_recovered) / true_intercept * 100
    print(f"Intercept       | {true_intercept:<10.2f} | {intercept_recovered:<10.4f} | {int_err:<10.2f}%")

    print("-" * 55)
    print(f"Final RMSE: {rmse_orig:.4f}")
    print(f"Noise Floor: {noise_std:.4f} (Ideal RMSE limit)")
    print(f"Performance: {'Good' if rmse_orig <= noise_std * 1.1 else 'Needs Tuning'}")
    print("="*30)
