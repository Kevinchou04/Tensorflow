import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    datalist=[]
    with open(filename) as fr:
        for line in fr.readlines():
            curline=line.strip().split('\t')
            fltline=list(map(float,curline))
            datalist.append(fltline)
    return datalist

def randCent(dataSet,k):
    n=np.shape(dataSet)[1]
    centroids=np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=np.mat(minJ+rangeJ*np.random.rand(k,1))
    return centroids

def kMeans(dataSet,k):
    m=np.shape(dataSet)[0]
    cluserAssment=np.mat(np.zeros((m,2)))
    centroids=randCent(dataSet,k)
    cluserChanged=True
    iterIndex=1
    while cluserChanged:
        cluserChanged=False
        for i in range(m):
            minDist=np.inf;minIndex=-1
            for j in range(k):
                distJI=np.linalg.norm(np.array(centroids[j,:])-np.array(dataSet[i,:]))
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if cluserAssment[i,0]!=minIndex:
                cluserChanged=True
            cluserAssment[i,:]=minIndex,minDist**2
            print("第%d次迭代后%d个质心的坐标：\n%s"%(iterIndex,k,centroids))
            iterIndex+=1
        for cent in range(k):
            ptsinclust=dataSet[np.nonzero(cluserAssment[:,0].A==cent)[0]]
            centroids[cent,:]=np.mean(ptsinclust,axis=0)
    return centroids,cluserAssment

def showCluster(dataSet,k,centroids,cluserAssment):
    numsamples,dim=dataSet.shape
    if dim!=2:
        return 1
    mark = ['or','ob','og','ok','oy','om','oc','^r','+r','sr','dr','<r','pr']
    for i in range(numsamples):
        markIndex=int(cluserAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    mark = ['Pr','Pb','Pg','Pk','Py','Pm','Pc','^b','+b','sb','db','<b','pb']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
    plt.show()

if __name__=='__main__':
    dataMat=np.mat(loadDataSet('dataSet.txt'))
    k=4
    cent,clust=kMeans(dataMat,k)
    showCluster(dataMat,k,cent,clust)