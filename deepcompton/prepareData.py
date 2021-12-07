from deepcompton.cones import AnglesDataset
import numpy as np
from sklearn.model_selection import train_test_split


def defineAllSample(dataTab):
    #nb of sample per line, i.e. for each simulated angle
    nbOfSample=20
    
    #maximum number of photons in each sample
    nbOfPhotonsMax=1000
    
    #minimum number of photons in each sample
    nbOfPhotonsMin=100

    srcTheta = []
    srcPhi = []

    trainAngles=[]
    testAngles=[]
    
    rowNumber = len(dataTab)
    for i in range(rowNumber):
      if i%100 ==0 :
        print(i)

      currentLine = dataTab[i]

      src_theta = currentLine[0]
      src_phi = currentLine[1]
      srcTheta.append(src_theta)
      srcPhi.append(src_phi)

      trainAngle , testAngle = createSample(currentLine, nbOfSample, nbOfPhotonsMin, nbOfPhotonsMax)

      trainAngles.append(trainAngle)
      testAngles.append(testAngle)

    return np.array(srcPhi), np.array(srcTheta), np.array(trainAngles), np.array(testAngles)

  
  
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



