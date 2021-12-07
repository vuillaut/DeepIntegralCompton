from deepcompton.cones import AnglesDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def defineAllSample(dataTab, nbOfSample=20, nbOfPhotonsMax=1000, nbOfPhotonsMin=100 ):
    """

    :param dataTab:
    :param nbOfSample: nb of sample per line, i.e. for each simulated angle
    :param nbOfPhotonsMax: maximum number of photons in each sample
    :param nbOfPhotonsMin: minimum number of photons in each sample
    :return: src_phi, src_theta, train_angles, test_angles
    """

    srcTheta = []
    srcPhi = []

    trainAngles=[]
    testAngles=[]

    for i, currentLine in enumerate(dataTab):
        mask = np.isfinite(currentLine['theta']) & np.isfinite(currentLine['phi']) & np.isfinite(currentLine['cotheta'])
        currentLine['theta'] = currentLine['theta'][mask]
        currentLine['phi'] = currentLine['phi'][mask]
        currentLine['cotheta'] = currentLine['cotheta'][mask]

        if i % 100 == 0:
            print(i)

        src_theta = currentLine[0]
        src_phi = currentLine[1]
        srcTheta.append(src_theta)
        srcPhi.append(src_phi)

        trainAngle, testAngle = createSample(currentLine, nbOfSample, nbOfPhotonsMin, nbOfPhotonsMax)

        trainAngles.append(trainAngle)
        testAngles.append(testAngle)

    trainAngles = np.array(trainAngles)
    trainAngles = trainAngles.reshape(trainAngles.shape[0] * trainAngles.shape[1], trainAngles.shape[3],
                                      trainAngles.shape[2])
    testAngles = np.array(testAngles)
    testAngles = testAngles.reshape(testAngles.shape[0] * testAngles.shape[1], testAngles.shape[3],
                                    testAngles.shape[2])

    srcTheta = np.deg2rad(np.repeat(srcTheta, nbOfSample))
    srcPhi = np.deg2rad(np.repeat(srcPhi, nbOfSample))

    return shuffle(srcPhi, srcTheta, trainAngles, testAngles)

  
  
def createSample(line,nbOfSample, nbOfPhotonsMin, nbOfPhotonsMax):

  trainTheta, testTheta, trainPhi, testPhi, trainCotheta, testCotheta = train_test_split(line['theta'],line['phi'],line['cotheta'], test_size=0.3)
  
  trainSamples= []
  testSamples = []

  for i in range(nbOfSample):

    sampledPhiTrain, sampledCothetaTrain, sampledThetaTrain = performUnitSample(nbOfPhotonsMin, nbOfPhotonsMax, trainTheta, trainCotheta, trainPhi)
    sampledPhiTest, sampledCothetaTest, sampledThetaTest = performUnitSample(nbOfPhotonsMin, nbOfPhotonsMax, testTheta, testCotheta, testPhi)

    resultTrain = np.array([ sampledPhiTrain, sampledCothetaTrain, sampledThetaTrain])
    resultTest = np.array([sampledPhiTest, sampledCothetaTest, sampledThetaTest])

    trainSamples.append(resultTrain)
    testSamples.append(resultTest)

  return np.array(trainSamples) ,  np.array(testSamples)



def performUnitSample(nbOfPhotonsMin, nbOfPhotonsMax, thetaVector, cothetaVector, phiVector):
  nbOfPhotons = np.random.randint(nbOfPhotonsMin,nbOfPhotonsMax)
  indexesAleatoires = np.random.randint(thetaVector.size, size = nbOfPhotons)

  sampledPhi = -0.99*np.ones(nbOfPhotonsMax)
  sampledCotheta = -0.99*np.ones(nbOfPhotonsMax)
  sampledTheta = -0.99*np.ones(nbOfPhotonsMax)
  
  sampledPhi[0:nbOfPhotons]=phiVector[indexesAleatoires]
  sampledCotheta[0:nbOfPhotons]= cothetaVector[indexesAleatoires]
  sampledTheta[0:nbOfPhotons]= thetaVector[indexesAleatoires]

  return np.array(sampledPhi), np.array(sampledCotheta), np.array(sampledTheta) 



