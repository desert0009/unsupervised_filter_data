from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class DIMENSION_REDUCTION_TSNE:
    def __init__(self, file_path, embeddings):
        self.file_path_list = file_path
        self.features = embeddings

    def t_sne(self, reduce_dim=2):
        tsne = TSNE(n_components=reduce_dim, learning_rate=120, perplexity=15, angle=0.2, verbose=2).fit_transform(self.features)

        # Normalize the embedding so that lies entirely in the range (0,1)
        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
        return tx, ty
    
    def draw(self, tx, ty, save_path):
        # Drawn t-SNE
        DRAW_W = 4000
        DRAW_H = 3000
        DRAW_MAX_DIM = 100 # The pixel size (on the largest size) to scale images to
        
        full_image = Image.new('RGBA', (DRAW_W, DRAW_H))
        for path, x, y in zip(self.file_path_list, tx, ty):
            tile = Image.open(path)
            rs = max(1, tile.width/DRAW_MAX_DIM, tile.height/DRAW_MAX_DIM)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((DRAW_W-DRAW_MAX_DIM)*x), int((DRAW_H-DRAW_MAX_DIM)*y)), mask=tile.convert('RGBA'))

        full_image.save(save_path)
