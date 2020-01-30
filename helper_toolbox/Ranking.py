import numpy as np

def ranking(current_image,images_features_scaled_pca,image_ids):
    score = []
    for image in images_features_scaled_pca[image_ids]:
        distance = np.linalg.norm(current_image-image)
        score.append(distance)

    ranklist = [x for _,x in sorted(zip(score,image_ids))]
    return ranklist
