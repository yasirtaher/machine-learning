import sys
sys.path.append("../01-GaussianNB_Deployment_on_Terrain_Data/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print submitAccuracy()