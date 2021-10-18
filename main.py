import argparse
import cv2
import os
import pickle
from tqdm import tqdm
from imutils import build_montages
import numpy as np
from sklearn.decomposition import PCA
from util import ob_detect, extract_embedding, tsne, kmeans

SAVE_PEOPLE = 'people_patches'
SAVE_EMBEDDING = 'people_embedding'
EMBEDDING_FILE = 'embeddings.p'
TSNE_FILE = 'tSNE.png'

PCA_N_COMPONENT = 128
K_MENAS_MIN = 10
K_MENAS_MAX = 30
RANDOM_SEED = 1223
COS_DIS_THR = 0.6

def crop_save_people(ob_model_path, data_path, save_path):
    # load object detect model
    ob_detector = ob_detect.ObjectDetect(ob_model_path)

    check_dir(save_path)

    # load img
    for dirPath, dirNames, fileNames in os.walk(data_path):
        for f in tqdm(fileNames):
            fn, ext = os.path.splitext(f)
            if ext not in ['.jpg', '.jpeg', '.png', '.JPG']:
                continue
            img_path = os.path.join(dirPath, f)
            img = read_img(img_path)
            img_h, img_w, _ = img.shape

            # object detection
            boxes, scores, classes = ob_detector.detect(img)
            people_boxes = ob_detector.get_people_boxes(boxes, scores, classes, conf_thr=0.65)

            # crop and save people patches
            for i, box in enumerate(people_boxes):
                y1, x1, y2, x2 = box
                y1, x1, y2, x2 = (int)(y1 * img_h), (int)(x1 * img_w), (int)(y2 * img_h), (int)(x2 * img_w)
                crop_img = img[y1:y2, x1:x2]
                save_file_name = '{}_{}_{}{}'.format(dirPath.replace('/', '_')[1:], fn, i, ext)
                cv2.imwrite(os.path.join(save_path, save_file_name), \
                            cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

def extract_save_embeddings(inp_path, save_path):
    check_dir(save_path)
    extractor = extract_embedding.FeatureExtractor()
    features, path_list = [], []
    for file in os.listdir(inp_path):
        path = os.path.join(inp_path, file)
        path_list.append(path)
        img = read_img(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Remove background color noise
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = extractor.data_preprocess(img)
        x = extractor.extract_embedding(x)
        x = extractor.reshape(x)
        features.append(x)
    pickle.dump([path_list, features], open(os.path.join(save_path, EMBEDDING_FILE), 'wb'))
    return path_list, features

def dimension_reduction(embedding_path, save_path):
    check_dir(save_path)
    path_list, features = pickle.load(open(embedding_path, 'rb'))
    dimension_reduction_tsne = tsne.DIMENSION_REDUCTION_TSNE(path_list, features)
    tx, ty = dimension_reduction_tsne.t_sne()
    dimension_reduction_tsne.draw(tx, ty, os.path.join(save_path, TSNE_FILE))

def cluster(embedding_path, save_path):
    # select num of cluster
    path_list, features = pickle.load(open(os.path.join(embedding_path, EMBEDDING_FILE), 'rb'))
    pca = PCA(n_components=PCA_N_COMPONENT)
    features = pca.fit_transform((features))
    k_menas = kmeans.K_MEAS(path_list, features)
    k_menas.search_num_cluster(features, K_MENAS_MIN, K_MENAS_MAX, RANDOM_SEED)
    print('select #cluster {}'.format(k_menas.k_start))

    # clustering
    k_menas.cluster(features, k_menas.k_start, 1223)
    
    # save results
    res = k_menas.cluster_filter(k_menas.data['cluster_result'], COS_DIS_THR, (260, 195))
    save_path = os.path.join(args.save_path, 'cluster')
    save_cluster_result(res, k_menas.k_start, save_path, (260, 195))

def save_cluster_result(res, k, save_path, resize_img_dim):
    for cid in range(k):
        res[cid].sort(key=lambda x: x[0]) # sort by cosine distance
        res[cid] = np.array(res[cid], dtype='object')
        imgs = res[cid][:, 1]
        dis_arr = res[cid][:, 0]
        avg_dis = sum(dis_arr) / len(dis_arr)
        
        montage = build_montages(imgs, resize_img_dim, (10, len(res[cid])//10 + 1))[0]
        check_dir(save_path)
        cv2.imwrite(os.path.join(args.save_path, 'cluster', '{}_avg{:.2f}.jpg').format(cid, avg_dis), montage)

def read_img(path):
    raw_img = cv2.imread(path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    return raw_img

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter Data')
    parser.add_argument('--ob_model_path', type=str, default='./model_zoo/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/')
    parser.add_argument('--data_path', type=str, default='./dataset/yoga/crawler')
    parser.add_argument('--save_path', type=str, default='./result/')
    args = parser.parse_args()

    # crop and save people patches
    print('Starting to crop the people patches via object detection ...')
    people_patches_path = os.path.join(args.save_path, SAVE_PEOPLE)
    crop_save_people(args.ob_model_path, args.data_path, people_patches_path)
    
    # extract and save embeddings
    print('Starting to extract embeddings ...')
    embedding_path = os.path.join(args.save_path, SAVE_EMBEDDING)
    _, _ = extract_save_embeddings(people_patches_path, embedding_path)

    # dimension reduction t-SNE
    # Note: we can reuse the features output of extract_save_embeddings()
    print('Dimension Reduction for Data Visualization t-SNE ...')
    tsne_save_path = os.path.join(args.save_path, 'tsne')
    dimension_reduction(os.path.join(embedding_path, EMBEDDING_FILE), tsne_save_path)

    # clustering k-means
    # define the num_of_cluster and clustering
    print('Clustering ...')
    save_path = os.path.join(args.save_path, 'cluster')
    cluster(embedding_path, save_path)
            

    