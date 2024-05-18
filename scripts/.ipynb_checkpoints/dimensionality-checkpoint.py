from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


def scaler(X):
    return (X-X.mean())/X.std()


def runPCA(X):
    # On pose le modèle
    X_scaled = scaler(X)
    pca = PCA()
    pca.fit(X_scaled)
    exp_var = pd.Series(pca.explained_variance_ratio_,name='ratio')*100
    
    print("="*100)
    print("PCA ratio de variance expliqué par composante")
    print((exp_var.to_frame())\
          		   .join(exp_var.cumsum().to_frame(name='cumsum_ratio')))
    print("="*100)
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(1,2,1)
    plt.title("PCA ratio de variance expliqué par composante")
    plt.plot(pca.explained_variance_ratio_.cumsum()*100,marker='o')
    exp_var.plot(kind="bar", alpha=1) # plot bar chart
    
    
    
    pca = PCA(2)
    pca.fit(X_scaled)
    
    plt.subplot(1,2,2)
    plt.title("Projection 2d PCA")
    
    X_transformed = pca.fit_transform(X_scaled)
    
    pca_x = X_transformed[:,0]
    pca_y = X_transformed[:,1]
    
    plt.scatter(pca_x,pca_y)
    
    plt.show()
    
    (fig, ax) = plt.subplots(figsize=(10,10))
    
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                     0,  # Start the arrow at the origin
                     pca.components_[0, i],  #0 for PC1
                     pca.components_[1, i],  #1 for PC2
                     head_width=0.1,
                     head_length=0.1)
        plt.text(pca.components_[0, i] + 0.05,
                     pca.components_[1, i] + 0.05,
                     X_scaled.columns.values[i])

    plt.xlabel(f'component 1 - {str(pca.explained_variance_ratio_[0]*100)[:4]} %')
    plt.ylabel(f'component 2 - {str(pca.explained_variance_ratio_[1]*100)[:4]} %')
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    ax.set_title('Cercle de corrélation : explication des composantes par facteur')
    plt.show()
    
    return pca_x,pca_y

