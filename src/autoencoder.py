from tabular_dae import DAE
from utils import *
import pandas as pd
from sklearn.preprocessing import StandardScaler


def autoencode():
    """
    Learn latent features via tabular autoencoder. Output the
    concatenated features to a new csv.
    """
    # Read data
    X_train, X_test, _ = load_data()
    X_train, X_test, _, _ = preprocess_data(X_train, X_test, _)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Combine the data
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    combined_df = pd.concat([X_test_df, X_train_df], axis=0)
    combined_df.reset_index(drop=True, inplace=True)

    # Fit the model on the combined data
    dae = DAE()
    dae.fit(combined_df, verbose=1)

    # Extract latent representation with the model
    train_latent = dae.transform(X_train_df)
    test_latent = dae.transform(X_test_df)

    # Combine the latent features with the original data
    latent_columns = [f"Latent{i}" for i in range(1, train_latent.shape[1] + 1)]
    train_latent = pd.DataFrame(train_latent, columns=latent_columns)
    test_latent = pd.DataFrame(test_latent, columns=latent_columns)

    # Save the combined data to a csv file
    train_latent.to_csv(TRAIN_LATENT_FN, index=False)
    test_latent.to_csv(TEST_LATENT_FN, index=False)


if __name__ == '__main__':
    autoencode()