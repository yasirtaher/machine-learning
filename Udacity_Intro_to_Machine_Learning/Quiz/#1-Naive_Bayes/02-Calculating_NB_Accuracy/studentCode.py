import sys
sys.path.append("../../common_files/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print submitAccuracy()