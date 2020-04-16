# simple code for hard c means
# written by rifkyfauzi9@gmail.com
import numpy as np 

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

def main():
    iter_num = 100
    cluster = 4
    error = 0.001
    data = np.array([  3.,    1.5,   2.,    0.5,   1.,  100.,    0.3,   2.1,   5.,    1.4,   0. ])
    np.random.shuffle(data)
    data_length = len(data)

    center = np.array([137.3062606,  951.4497057,  817.17942654,  27.86459199]) #
    # np.random.seed(4)
    # center = np.random.rand(cluster)*1000 #


    matrix = np.zeros((data_length, cluster)) #baris, kolom
    distances = np.zeros((data_length)) 
    print(data, center)



    # main function 
    distances = np.zeros((data_length, cluster)) 
    idx = np.zeros(cluster)
    epsilon = 10
    iter = 0

    for n in range(2):
        
        for j in range(cluster):  #loop in cluster center
            for i in range(data_length):
                    distances[i, j] = abs(data[i] - center[j])
            
        matrix = membership(distances)
        print('=================================================')
        print(matrix)
        print('=================================================')
        # print(distances)
        # print('=================================================')


        
        member_num = np.zeros(cluster)
        for j in range(cluster):
            member_num[j] = np.sum(matrix[:,j])


        ################ additional feature to encounter isolated center(s)#############################################
        if min(member_num) == 0: # isolated center with no member
            print('ada')
            temp = -1
            #assigment based on priority k = 0,1,2,..
            member_priority = index_priority(member_num)
            # print(member_priority)
            # print(member_priority[:-1])
            # print('=================================================')
            print(member_priority)
            k = 0 #priority number 1
            single = []
            for p in member_priority[:-1]:
                if np.sum(matrix[:,p]) == 1:
                    single.append(p)
            print('non-empty {}'.format(single))
            for p in member_priority[:-1]:
                if np.sum(matrix[:,p]) == 0:
                    center_priority = index_priority(distances[:,p])
                    print(center_priority)
                    if not single:
                        print("non-empty")
                    if center_priority[k] != temp:
                        data_index = center_priority[k]
                        if data_index == single[k]:
                            data_index = center_priority[k+1]
                    else:  
                        data_index = center_priority[k+1]


                    a = member_priority[-1] #location index for center with many member
                    m = find_min_array(distances[:,p]) #minimum distance index for center for every data
                    matrix[ data_index , p ] = 1
                    matrix[ data_index, a ] = 0
                    print('k = {}'.format(k))
                    print('matrix')
                    print(matrix)
                    k = k + 1
                    temp = data_index
                    

        ################ additional feature to encounter isolated center(s)#############################################
        
        new_center = np.zeros(cluster)
        for j in range(cluster):
            new_center[j] = np.dot(data, matrix[:,j])/np.sum(matrix[:,j])

        epsilon = np.linalg.norm(center - new_center)
        center = new_center
        iter += 1

        # just for safety 
        if iter > 10^5:
            break

    return center


not_same = True
while not_same == True:
    print("=======================================Start===================================================")
    center = main()
    unique = []
    for c in center:
        if c not in unique:
            unique.append(c)

    if len(unique) != len(center):
        print("yea")
        not_same = False
    else:
        not_same = True
    



print(center)
