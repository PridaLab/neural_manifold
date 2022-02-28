import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances


def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    nnDist = np.sum(D < np.percentile(D,1), axis=1) - 1
    data = np.delete(data, np.where(nnDist < np.percentile(nnDist, 20))[0], axis=0)
    return data



def generate_graph_laplacian(data, nn):
    """Generate graph Laplacian from data."""
    # Adjacency Matrix.
    connectivity = kneighbors_graph(X=data, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1/2)*(connectivity + connectivity.T)
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian


def compute_spectrum_graph_laplacian(graph_laplacian):
    """Compute eigenvalues and eigenvectors and project 
    them onto the real numbers.
    """
    if graph_laplacian.shape[0] > 50:
        kvec = 50
    else:
        kvec = graph_laplacian.shape[0] - 1
    if np.all(np.asmatrix(graph_laplacian).H == graph_laplacian):
        eigenvals, eigenvcts = sparse.linalg.eigsh(graph_laplacian, k=kvec, which='SM')
    else:
        eigenvals, eigenvcts = sparse.linalg.eigs(graph_laplacian, k=kvec, which='SM')
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts


def compute_connected_components(data):
    if data.shape[0] >= 20:
        data = filter_noisy_outliers(data)
        graph_laplacian = generate_graph_laplacian(data, 8)
        eigenvals, eigenvcts = compute_spectrum_graph_laplacian(graph_laplacian)
        kn = len(np.argwhere(abs(eigenvals) < 1e-5))
    else:
        kn = 1
    return kn


def assign_labels(data, kn):
    if bool(kn) and kn > 1:
        clusModel = SpectralClustering(n_clusters=kn, affinity='nearest_neighbors', assign_labels='kmeans')
        clusModel.fit(data)
        labels = clusModel.labels_ + 1
    else:
        labels = np.zeros(data.shape[0]) + 1
    return labels


def compute_pointCloudsOverlap(cloud1, cloud2, k):
    breakpoint()
    #Stack both clouds
    cloud_all = np.vstack((cloud1, cloud2))
    #Create cloud label
    cloud_label = np.hstack((np.ones(cloud1.shape[0]), np.ones(cloud2.shape[0])*2))
    #Compute k neighbours graph wieghted by cloud label
    connectivity = kneighbors_graph(X=cloud_all, n_neighbors=k, mode='connectivity', include_self = False).toarray() * cloud_label
    #Compute the degree of each point of cloud 1
    degree = np.sum(connectivity, axis=1)[cloud_label==1]
    #Compute overlap percentage
    overlap = (np.sum(degree > k) / degree.shape[0])*100
    return overlap



def compute_structure_index(emb, label, nBins, dimNames, plotCluster, overlapThreshold=0.5, **kwargs):
    #Preprocess data 
    emb = emb[:,dimNames]
    for d in range(emb.shape[1]):
        emb[:,d] = (emb[:,d] - np.nanmean(emb[:,d])) / np.nanstd(emb[:,d])
    #Delete nan values from label and emb
    emb = np.delete(emb, np.where(np.isnan(label))[0], axis=0)
    label = np.delete(label, np.where(np.isnan(label))[0], axis=0)
    #If there is a predefined max or min delete all points out of bounds
    if 'vmin' in kwargs:
        #emb = np.delete(emb, np.where(label<kwargs['vmin'])[0], axis=0)
        label[np.where(label<kwargs['vmin'])[0]] = kwargs['vmin']
        #label = np.delete(label, np.where(label<kwargs['vmin'])[0], axis=0)
    if 'vmax' in kwargs:
        #emb = np.delete(emb, np.where(label>kwargs['vmax'])[0], axis=0)
        label[np.where(label>kwargs['vmax'])[0]] = kwargs['vmax']
        #label = np.delete(label, np.where(label>kwargs['vmax'])[0], axis=0)
    #Create the bin edges
    if 'vmin' in kwargs and 'vmax' in kwargs:
        binSize = (kwargs['vmax'] - kwargs['vmin']) / nBins
        binEdges = np.column_stack((np.linspace(kwargs['vmin'],kwargs['vmin']+binSize*(nBins-1),nBins),
                                    np.linspace(kwargs['vmin'],kwargs['vmin']+binSize*(nBins-1),nBins)+binSize))
    else:
        binSize = (np.max(label) - np.min(label)) / nBins
        binEdges = np.column_stack((np.linspace(np.min(label),np.min(label)+binSize*(nBins-1),nBins),
                                    np.linspace(np.min(label),np.min(label)+binSize*(nBins-1),nBins)+binSize))

    #Create binLabel
    binLabel = np.zeros(label.shape)
    for b in range(nBins-1):
        binLabel[np.logical_and(label >= binEdges[b,0], label<binEdges[b,1])] = 1 + int(np.max(binLabel))
    binLabel[np.logical_and(label >= binEdges[nBins-1,0], label<=binEdges[nBins-1,1])] = 1 + int(np.max(binLabel))

    #Discard outlier clusters (nPoints < 1%)
    #Compute number of points in each cluster
    nPoints = np.array([np.sum(binLabel==value) for value in np.unique(binLabel)])
    #Get the clusters that meet criteria and delete them
    delLabels = np.where(nPoints < label.size*1/100)[0]
    #Delete outlier clusters
    for delInd in delLabels:
        binLabel[binLabel==delInd+1] = 0
    #Renumber bin labels from 1 to n clusters
    uniqueVal = np.unique(binLabel)
    if 0 in np.unique(binLabel):
        for idx in range(1,len(uniqueVal)):
            binLabel[binLabel==uniqueVal[idx]]= idx

    #Compute the cluster index
    #Compute overlap between clusters pairwise
    overlapMat = np.zeros((np.sum(np.unique(binLabel) > 0), np.sum(np.unique(binLabel) > 0)))
    for ii in range(overlapMat.shape[0]):
        for jj in range(overlapMat.shape[1]):
            if ii != jj:
                overlap = compute_pointCloudsOverlap(emb[binLabel==ii+1,:], emb[binLabel==jj+1,:], 3)
                overlapMat[ii,jj] = overlap/100
    #Symetrize overlap matrix
    overlapMat = (overlapMat + overlapMat.T) / 2
    clusterIndex = 1 - np.mean(np.sum(1*(overlapMat>=overlapThreshold), axis=0))/(overlapMat.shape[0]-1)

    return clusterIndex, binLabel, overlapMat