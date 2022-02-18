import scipy
import scipy.io
import umap
import numpy as np
import matplotlib.pyplot as plt


def loadMatFile(filePath):
	file = scipy.io.loadmat(filePath)
	data = file[list(file.keys())[-1]]
	return data


def umapReduction(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', metric_kwds=None):
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        metric_kwds=metric_kwds
    )
    embedding.fit(data)
    return embedding


def transformEmbedding(data, p1, p2, visualCheck):
    #Compute mid point between p1 and p2
    mid = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
    #Translate dataset to match mid with origin 
    translation = np.array([0-mid[0], 0-mid[1]])
    p1t = p1.copy() + translation
    p2t = p2.copy() + translation
    midt = mid.copy() + translation
    #Get the highest centroid
    if p1[1] > p2[1]:
        p = p1t
    else:
        p = p2t
    #Compute the angle between the point segment and the horizontal segment
    alpha = np.arccos(np.abs(p[0])/np.sqrt(p[0]**2 + p[1]**2)) #In radians
    #Check if rotate counter or clockwise and create rotation matrix
    if p[0] > 0 and p[1] > 0: #clockwise (alpha +)
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
    elif p[0] < 0 and p[1] > 0: #counterclockwise (alpha -)
        R = np.array([[np.cos(-alpha), -np.sin(-alpha)],
                      [np.sin(-alpha), np.cos(-alpha)]])
    #Rotate coordinates
    p1tr = np.dot(p1t,R)
    p2tr = np.dot(p2t,R)
    midtr = np.dot(midt,R)
    #Transform all data
    datat = data + translation
    datatr = np.empty(datat.shape)
    for ii in range(data.shape[0]):
        datatr[ii] = np.dot(datat[ii], R)
    #Project the data on the horizontal line
    projection = datatr[:,0]
    #Visual check
    if visualCheck:
        f1, axs = plt.subplots(1, 3, figsize=(15,10.5))
        #Before transformation
        axs[0].scatter(data[:,0], data[:,1], c='k', s=10, alpha=0.4)
        axs[0].scatter(p1[0], p1[1], c='r', s=50)
        axs[0].scatter(p2[0], p2[1], c='b', s=50)
        axs[0].scatter(mid[0], mid[1], c='g', s=50)
        axs[0].axvline(x=0)
        axs[0].axhline(y=0)
        axs[0].set_title('Before transformation')
        #After translation
        axs[1].scatter(datat[:,0], datat[:,1], c='k', s=10, alpha=0.4)
        axs[1].scatter(p1t[0], p1t[1], c='r', s=50)
        axs[1].scatter(p2t[0], p2t[1], c='b', s=50)
        axs[1].scatter(midt[0], midt[1], c='g', s=50)
        axs[1].axvline(x=0)
        axs[1].axhline(y=0)
        axs[1].set_title('After translation')
        #After rotation
        axs[2].scatter(datatr[:,0], datatr[:,1], c='k', s=10, alpha=0.4)
        axs[2].scatter(p1tr[0], p1tr[1], c='r', s=50)
        axs[2].scatter(p2tr[0], p2tr[1], c='b', s=50)
        axs[2].scatter(midtr[0], midtr[1], c='g', s=50)
        axs[2].axvline(x=0)
        axs[2].axhline(y=0)
        axs[2].set_title('After rotation')
        #Set all lims equal
        xlims = np.array([axs[0].get_xlim(), axs[1].get_xlim(), axs[2].get_xlim()])
        xL = [np.min(xlims[:,0]), np.max(xlims[:,1])]
        ylims = np.array([axs[0].get_ylim(), axs[1].get_ylim(), axs[2].get_ylim()])
        yL = [np.min(ylims[:,0]), np.max(xlims[:,1])]
        axs[0].set_xlim(xL)
        axs[1].set_xlim(xL)
        axs[2].set_xlim(xL)
        axs[0].set_ylim(yL)
        axs[1].set_ylim(yL)
        axs[2].set_ylim(yL)
    return projection
