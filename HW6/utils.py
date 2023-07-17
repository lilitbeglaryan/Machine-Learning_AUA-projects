"""
Insert your utility functions in this module
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from timeit import default_timer as timer



def load_faces():  # you can give other arguments if you wish
    # do not modify `fetch_lfw_people` default arguments
    faces_dataset = fetch_lfw_people(resize=0.7, min_faces_per_person=50, color=False)
    return (faces_dataset.data, faces_dataset.images, faces_dataset.target, faces_dataset.target_names)


def plot_pca_faces(X, image_shape, n_comp):
    X_back = pca_faces(X,n_comp)

    # plot the first three images in the test set:
    fix, axes = plt.subplots(10, 2, figsize=(8, 35),subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in enumerate(axes):
        # plot original image
        ax[0].imshow(X[i].reshape(image_shape),vmin=0, vmax=1, cmap="gray", aspect = "auto")  #original image
        
        # plot the four back-transformed images
        ax[1].imshow(X_back[i].reshape(image_shape), vmin=0, vmax=1, cmap="gray", aspect = "auto")
        if i == 0:
          ax[0].set_title("original image")
          ax[1].set_title("reduced to %d component" % n_comp)
    plt.show()
    # label the top row
    # axes[0].set_title("original image")
   
    # ax.set_title("reduced to %d component" % n_comp)
    # plt.tight_layout()

def pca_faces(X, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    X_pca= pca.transform(X)
    X_back = pca.inverse_transform(X_pca)
    return X_back

def PCA_(train, test,n_comp=None):
  pca = PCA(n_components=n_comp,whiten=True, random_state=42).fit(train)
  X_pca = pca.transform(train)
  X_tst_pca = pca.transform(test)
  print("X_train_pca.shape: {}".format(X_pca.shape))
  return(X_pca, X_tst_pca, pca)

def model_duration(model,X_tr,X_tst, y_tr,inference=True):
  if(inference):
    duration = []
    for i in range(100):
      start = timer()
      model.predict(X_tr)
      duration.append(timer() - start)
    return sum(duration)/len(duration)
  else:
    start = timer()
    # model = RandomForestClassifier(criterion = best_params["criterion"], max_depth = best_params["max_depth"], max_features = best_params["max_features"])
    model.fit(X_tr,y_tr)
    return timer()-start