import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2

def load_images_from_directory(directory):
    y = []  # Labels
    X = []  # Flattened images
    target_names = []  # Person names
    person_id = 0
    h, w = 300, 300  # Image dimensions

    for person_name in os.listdir(directory):
        dir_path = os.path.join(directory, person_name)
        target_names.append(person_name)

        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (h, w))
            flattened_image = resized_image.flatten()
            
            X.append(flattened_image)
            y.append(person_id)

        person_id += 1

    return np.array(X), np.array(y), np.array(target_names)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def main():
    dir_name = "Desktop/dataset/faces/"
    X, y, target_names = load_images_from_directory(dir_name)
    
    n_samples, n_features = X.shape
    print("Number of samples:", n_samples)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # PCA
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, 300, 300))
    eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, 300, 300)
    plt.show()

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    # MLP Training
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True)
    clf.fit(X_train_lda, y_train)

    # Prediction and Evaluation
    y_pred = clf.predict(X_test_lda)
    accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
    print("Accuracy:", accuracy)

    # Visualization
    prediction_titles = [f"pred: {target_names[pred]}, true: {target_names[true]}" for pred, true in zip(y_pred, y_test)]
    plot_gallery(X_test, prediction_titles, 300, 300)
    plt.show()

if __name__ == "__main__":
    main()
