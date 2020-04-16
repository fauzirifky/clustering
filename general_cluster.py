# simple code for hard c means
# written by rifkyfauzi9@gmail.com
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colors = ['red', 'blue', 'green', 'orange', 'indigo']
iter = 100
cluster = 4
error = 0.001
np.random.seed(200000)

# data = np.array([[1,1],[2,1],[4,3],[5,4]])
# data = np.array([[0.5, 0.0], [2.0, 5.0], [25, 1.0], [0, 13], [1.5, 90], [1.4, 2], [0.3, 14], [2.1, 43], [3, 0], [200,100]])
# data = np.array([0.5, 2.0, 1.0, 0, 1.5, 1.4, 0.3, 2.1, 3, 200])
data = np.random.rand(100, 2)

try:
    data_length, dim = data.shape
except:
    data_length = len(data)
    dim = 1

if cluster > data_length:
    print('number of cluster must be less than data length')
    cluster = data_length
center = np.random.rand(cluster, dim)#np.random.rand(cluster) #



matrix = np.zeros((data_length, cluster)) #baris, kolom
distances = np.zeros((data_length)) 

def find_min_array(distance):
    N = len(distance)
    a = min(distance)
    for i in range(N):
        if distance[i] == a:
            idx = i
    return idx #location and value



def membership(distances):

    N, cluster = distances.shape
    matrix = np.zeros((N, cluster))
    
    for i in range(N): #cluster
        a = min(distances[i,:])
        for j in range(cluster): #data
            if distances[i,j] == a:
                matrix[i,j] = 1
                
            else:
                matrix[i,j] = 0
    return matrix

def index_priority(array):
    a = [i[0] for i in sorted(enumerate(array), key=lambda x:x[1])]
    return a


        

# main function 
distances = np.zeros((data_length, cluster)) 
idx = np.zeros(cluster)

for n in range(iter):
    
    for j in range(cluster):  #loop in cluster center

        for i in range(data_length):
                distances[i, j] = np.linalg.norm(data[i,:] - center[j,:])
        

    matrix = membership(distances)
    isolated_indexes = []
    minimum_distances = []

    
    priority = np.zeros((data_length, cluster))
    for j in range(cluster):
        priority[:,j] = index_priority(distances[:,j])
    # print(priority)
    
    ################ additional feature to encounter isolated point(s)
    #assigment based on priority k = 0,1,2,..
    k = 0 #priority number 1
    for j in range(cluster):
        if np.sum(matrix[:,j]) == 1:
            k = k + 1 #update priority, cluster with one element cannot be recalculated
        if np.sum(matrix[:,j]) == 0:
            p = priority[:,j]
            a = int(p[k]) 
            m = find_min_array(distances[a,:])
            matrix[ a , j ] = 1
            matrix[ a , m ] = 0
            
    
    new_center = np.zeros((cluster,dim))

    for j in range(cluster):
        for d in range(dim):
            new_center[j, d] = np.dot(data[:, d], matrix[:,j])/np.sum(matrix[:,j])

   

    
    # set category for visualization
    cat = {}
    for j in range(cluster):
        cat[j] = []

    for i in range(data_length):
        idx = find_min_array(-1*matrix[i,:]) 
        cat[idx].append(data[i, :])
    
    
    #reshaping data set
    for j in range(cluster):
        N = len(cat[j])
        cat[j] = np.array(cat[j])
       
        
    fig = plt.figure()
    if dim == 2:
        
        for j in range(cluster):
            plt.scatter( cat[j][:,0], cat[j][:,1], color = colors[j])
            plt.scatter( center[j,0], center[j,1], marker = '^', color = 'black' )
            for i in range(len(cat[j])):
                plt.plot([center[j,0], cat[j][i,0]], [center[j,1], cat[j][i,1]], ':', color = colors[j])
    else:
         
         ax = fig.add_subplot(111, projection='3d')
         for j in range(cluster):
            ax.scatter( cat[j][:,0], cat[j][:,1], cat[j][:,2], color = colors[j])
            ax.scatter( center[j,0], center[j,1], center[j,2], marker = '^', color = 'black' )
            ax.view_init(30, 30 + n*1)
            for i in range(len(cat[j])):
                plt.plot([center[j,0], cat[j][i,0]], [center[j,1], cat[j][i,1]], [center[j,2], cat[j][i,2]], ':', color = colors[j])       
        
    plt.pause(0.1)
    plt.close()
    
    center = new_center
    
plt.show(block=False) 


print("========================")
print(new_center)
