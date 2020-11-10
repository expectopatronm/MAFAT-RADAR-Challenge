from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np

def stats(pred, actual):
  plt.figure(figsize = (20, 10))
  fpr1, tpr1, _ = roc_curve(actual[0], pred[0])
  fpr2, tpr2, _ = roc_curve(actual[1], pred[1])
  roc_auc = [auc(fpr1, tpr1), auc(fpr2, tpr2)]
  lw = 2
  plt.plot(fpr1, tpr1, lw=lw, label='Training set (ROC-AUC = %0.4f)' % roc_auc[0])
  plt.plot(fpr2, tpr2, lw=lw, label='Validation set (ROC-AUC = %0.4f)' % roc_auc[1])
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize = 18)
  plt.ylabel('True Positive Rate', fontsize = 18)
  plt.title('Training set vs. Validation set ROC curves')
  plt.legend(loc = "lower right", prop = {'size': 20})
  plt.show()

def feature_viewer(model, image):
  
  plt.imshow(np.squeeze(image), cmap = 'viridis')
  
  successive_outputs = [layer.output for layer in model.layers[1:]]

  visualization_model = Model(inputs = model.input, outputs = successive_outputs)

  x = image
  x = x.reshape((1,) + x.shape)             

  successive_feature_maps = visualization_model.predict(x)

  layer_names = [layer.name for layer in model.layers]

  for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    
    if len(feature_map.shape) == 4:

      n_features = feature_map.shape[-1]  
      sizex = feature_map.shape[1] 
      sizey = feature_map.shape[2] 
      display_grid = np.zeros((sizex, sizey * n_features))
      
      for i in range(n_features):
        x = feature_map[0, :, :, i]
        x -= x.mean()
        x /= x.std()
        x *= 64
        x += 128
        x = np.clip(x, 0, 255).astype('uint8')
        display_grid[:,i*sizey:(i+1)*sizey] = x

      scale = 25./n_features
      plt.figure(figsize=(scale * n_features, scale), dpi = 300)
      plt.title(layer_name)
      plt.grid(False)
      plt.imshow( display_grid, aspect = 'auto', cmap = 'viridis')