# simple code for hard c means
# written by rifkyfauzi9@gmail.com
import numpy as np 

iter = 10000
cluster = 4
error = 0.001
data = np.array([0.5, 2.0, 1.0, 0])
data_length = len(data)
center = np.random.rand(cluster) #

matrix = np.zeros((data_length,cluster)) #baris, kolom
distances = np.zeros((data_length,cluster)) 

def find_min_array(distance):
    N = len(distance)
    a = min(distance)
    for i in range(N):
        if distance[i] == a:
            idx = i
    return idx, a #location and value


print('center', center)
def membership(idx, data_length, cluster):
    matrix = np.zeros((data_length,cluster))
    for i in range(data_length):
        for j in range(cluster):
            if idx[i] == j:
                matrix[i,j] = 1
            else:
                matrix[i,j] = 0
    return matrix

# main function 
matrix = np.zeros((data_length,cluster)) 
for n in range(iter):
    
    for j in range(cluster):  

        while np.sum(matrix[:,j]) == 0:
            center[j] = np.random.choice(data)
            for i in range(data_length):
                for j in range(cluster):
                    distances[i,j] = abs(data[i] - center[j])
            idx = np.zeros(data_length); value = np.zeros(data_length)

            for i in range(data_length):
                idx[i], value[i] = find_min_array(distances[i,:])


            matrix = membership(idx, data_length, cluster)


    new_center = np.zeros(cluster)
    for j in range(cluster):
        new_center[j] = np.dot(data, matrix[:,j])/np.sum(matrix[:,j])


    center = new_center
print('matrix', matrix)
print(' center : \n',new_center)
print('iter',n)
