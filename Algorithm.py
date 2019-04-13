
import matplotlib.pyplot as plt
import Node
import pandas
from sklearn.cluster import KMeans
import math
import Cluster
import networkx as nx
from dijkstra import Graph
import numpy as np
import random
import datetime


# Given information
numNodes = 300
xmin = 0
ymin = 0
xmax = 10000
ymax = 10000
Receiver_sensitivity = -89 ##dBm
Transmission_power = +14   ##dBm
Tx_Rx_antenna_gain = 1
frequency = (2400)
cluster = 18
No_of_cluster_movement = 0
no_of_new_cluster_added = 0
Dictionary_power = {}
transection_range = 1000

# Declaring a list of object (node)
nodes = list()
node_number_list = []
print("github")

starttime = datetime.datetime.now()

#path loss calculation from transmission power and receiver sensitivity
Tx_power_budget = (Transmission_power + Tx_Rx_antenna_gain)
PathLoss = Tx_power_budget - Receiver_sensitivity
Distance_DB = PathLoss- 36.56 - 20*math.log10(frequency)
Distance_miles = 10**(Distance_DB/20)
Distance = Distance_miles*(1609.34)

# Creating a list of nodes with coordinate,id
for nodeid in range(numNodes):
    new_obj= Node.Node(nodeid, xmax, ymax, xmin, ymin)
    nodes.append(new_obj)
    node_number_list.append(nodeid)

#Create a pandas dataset
data = pandas.DataFrame.from_records([s.to_dict() for s in nodes])

#Plot the nodes
plt.figure(1)
plt.title('Nodes_Clusters')
plt.scatter(data['x_axis'], data['y_axis'],label='Node')
plt.legend()

# keeping coordinate of noes into another dataset for advantage of using
dataset= data[['x_axis','y_axis']]

# Applying KMeans algorithm on the random nodes using dataset
kmeans = KMeans(cluster)
kmeans.fit(dataset)
labels = kmeans.predict(dataset)
centroid = kmeans.cluster_centers_

# plotting the centroids into figure (1)
plt.scatter(centroid[:,0],centroid[:,1], s=150, marker='D', c='red', label='cluster_Head')
plt.legend()


#list of original centroid
list_centroid = {}
Cluster_Information = []
cluster_count = -1

# Drawing circle around centroid
for center in centroid:
    cluster_count = cluster_count + 1
    x = center[0]
    y = center[1]
    circle = plt.Circle((x,y),Distance, color='g', fill=False)
    plt.gcf().gca().add_artist(circle)
    list_centroid[cluster_count] = (x,y, Distance)
    cluster_info = Cluster.Cluster(cluster_count, x, y)
    Cluster_Information.append(cluster_info)
    dcol= 'C' + str(cluster_count) + 'in'
    Dictionary_power[dcol] = Transmission_power

# Making a copy of centroid dictionarey for the advantage of better use
Revised_centroid_list = list_centroid

# Adding lable column in pandas dataframe and keep the labels into that column
label = pandas.DataFrame(labels, columns=['labels'])
data = data.join(label)

# Methods for Ceating function for best cluster movement optimization
def newcluster_inside(newx, newy, radi):
    inside_list=[]
    for enode in nodes:
        inside1 = False
        distance = 0
        distance = math.sqrt(((newx - enode.x_axis) ** 2) + ((newy) - enode.y_axis) ** 2)
        if distance < radi:
            inside1 = True
        inside_list.append(inside1)
    return inside_list

# Methods for Replacement of centroid if it moves
def centroid_repalcement(nearestx, nearesty, newcirclex, newcircley, Center_List):
    revised_centroid_list =Center_List
    #revised_centroid_list[]
    bal =[]
    man = 0
    for i in range(len(Center_List)):
        if Center_List[i] == (nearestx, nearesty):
            revised_centroid_list[i] = (newcirclex, newcircley)
            man = i
            bal = revised_centroid_list
    return bal, man

# Methods for Clusterwise nodelist and distance providing
def ClusterNodeDist(maa, baba, nodes, Distance):
    clusterwise_nodelist = []
    max_distance = 0
    for pi in nodes:
        check = []
        distancee = math.sqrt(((pi.x_axis - maa) ** 2) + ((pi.y_axis - baba) ** 2))
        if distancee <= Distance:
            check.append(distancee)
            check.append(pi.x_axis)
            check.append(pi.y_axis)
            mytuple = tuple(check)
            clusterwise_nodelist.append(mytuple)
            if distancee > max_distance:
                max_distance = distancee
    return clusterwise_nodelist, max_distance

##### First of all, for each node check if it is inside a circle or outside. Distance of a node from each centorid.
##### find out the nearest cluster for each node and distance form that cluster.
##### Secondly, Check a node is orphan or not. if orphan then shift the nearest cluster so that it  can cover that orphan node.
##### Check if moving this cluster makes more node orphan then it will node move rather, it will create new cluster around the child node of a closest cluster that is nearest to the orphapns nodes with smaller transmission range

possible_move_centroid =[]
New_Center = []
col_increment = 0
Total_Cluster = cluster + no_of_new_cluster_added

for eachnode in nodes:
    col_increment = -1
    check = xmax + ymax
    closest_cluster = ''
    closest_cluster_id = 0

    for key, eachcenter in Revised_centroid_list.items():
        col_increment = col_increment+1
        inside, distance = eachnode.NodeClusterDist(eachcenter[0], eachcenter[1], Distance)
        icol = 'C'+str(col_increment) + 'in'
        dcol = 'C'+str(col_increment) + 'dis'
        data.loc[eachnode.id, dcol] = distance
        data.loc[eachnode.id, icol] = inside
        if distance<check:
            check = distance
            closest_cluster = eachcenter
            closest_cluster_id = key
    data.loc[eachnode.id, 'nearest_cluster_dist'] = check
    data.loc[eachnode.id, 'nearest_cluster_xaxis'] = closest_cluster[0]
    data.loc[eachnode.id, 'nearest_cluster_yaxis'] = closest_cluster[1]
    data.loc[eachnode.id, 'nearest_cluster_id'] = closest_cluster_id
    data.loc[eachnode.id, 'orfan'] = sum(data.loc[eachnode.id, ['C' + str(i) + 'in' for i in range(cluster)]])

    if data.loc[eachnode.id,'orfan'] <= 0:
        keep_x = data.loc[eachnode.id,'x_axis'] - data.loc[eachnode.id,'nearest_cluster_xaxis']
        keep_y = data.loc[eachnode.id,'y_axis'] - data.loc[eachnode.id,'nearest_cluster_yaxis']
        node_cluster_distance = data.loc[eachnode.id,'nearest_cluster_dist']- Distance
        ratio = (node_cluster_distance + 0.03) / data.loc[eachnode.id,'nearest_cluster_dist']
        new_cluster_xaxis = data.loc[eachnode.id, 'nearest_cluster_xaxis'] + ratio*keep_x
        new_cluster_yaxis = data.loc[eachnode.id, 'nearest_cluster_yaxis'] + ratio * keep_y

        data.loc[eachnode.id, 'new_cluster_xaxis'] = new_cluster_xaxis
        data.loc[eachnode.id, 'new_cluster_yaxis'] = new_cluster_yaxis
        possible_move_centroid.append((new_cluster_xaxis, new_cluster_yaxis))

        inside_list1 = newcluster_inside(new_cluster_xaxis, new_cluster_yaxis, Distance)
        xxx = data.loc[eachnode.id,'nearest_cluster_xaxis']
        yyy = data.loc[eachnode.id,'nearest_cluster_yaxis']
        cid = data.loc[eachnode.id, 'nearest_cluster_id']
        inside_list2 = newcluster_inside(xxx, yyy, Distance)

        aa = sum(inside_list1)
        bb = sum(inside_list2)

        if aa > bb:
            No_of_cluster_movement = No_of_cluster_movement + 1
            Revised_centroid_list[cid] = (new_cluster_xaxis, new_cluster_yaxis, Distance)

        else:
            mim = data.loc[eachnode.id, 'x_axis']
            kim = data.loc[eachnode.id, 'y_axis']
            clusterwise_nodelist, max_distance = ClusterNodeDist(xxx, yyy, nodes, Distance)
            reasonable_distance = 10000
            NEWC_X = 0
            NEWC_Y = 0
            for dddist in clusterwise_nodelist:
                node2node_distance = math.sqrt(((mim - dddist[1]) ** 2) + ((kim - dddist[2]) ** 2))
                if node2node_distance < reasonable_distance:
                    reasonable_distance = node2node_distance
                    NEWC_X = dddist[1]
                    NEWC_Y = dddist[2]
            reasonable_distance = reasonable_distance + 0.03
            New_Center.append((NEWC_X, NEWC_Y, reasonable_distance))

            #calculating new TX power
            reasonable_distance_miles = 0.000621371 * reasonable_distance
            if(reasonable_distance_miles <= 0):
                reasonable_distance_miles = 0.001
            reasonable_distance_DB = 20 * math.log10(reasonable_distance_miles)
            PathLoss_DB = 36.56 + 20 * math.log10(frequency) + reasonable_distance_DB
            new_cluster_Txp_with_gain = PathLoss_DB + Receiver_sensitivity
            new_cluster_Txp = new_cluster_Txp_with_gain - Tx_Rx_antenna_gain
            new_cluster_Txp = round(new_cluster_Txp)

            no_of_new_cluster_added = no_of_new_cluster_added + 1
            col_stat = 'C' + str(Total_Cluster - 1) + 'in'
            Dictionary_power[col_stat] = new_cluster_Txp

            Revised_centroid_list[no_of_new_cluster_added + cluster - 1] = (NEWC_X, NEWC_Y,reasonable_distance)

            for eachnode1 in nodes:
                check = xmax + ymax
                closest_cluster1 = ''
                for eachcenter1 in New_Center:
                    inside, distance = eachnode1.NodeClusterDist(eachcenter1[0], eachcenter1[1], eachcenter1[2])
                    dcol1 = 'C' + str(Total_Cluster - 1) + 'dis'
                    data.loc[eachnode1.id, col_stat] = inside
                    data.loc[eachnode1.id, dcol1] = distance
                data.loc[eachnode1.id, 'Revised-orfan'] = sum(
                    data.loc[eachnode1.id, ['C' + str(i) + 'in' for i in range(Total_Cluster)]])

    else:
        data.loc[eachnode.id, 'new_cluster_xaxis']= "Not-a-Orphan"
        data.loc[eachnode.id, 'new_cluster_yaxis']= "Not-a-Orphan"

# Printing total number of cluster
print('Total no of cluster: ', Total_Cluster)

# plotting of cluster after movement, addition and shrink
plt.figure(2)
plt.title('Clusters After movement')
plt.scatter(data['x_axis'], data['y_axis'],label='Node')
plt.legend()
for key, center in Revised_centroid_list.items():
    x = center[0]
    y = center[1]
    Radius = center[2]
    plt.scatter(x, y, s=150, marker='*', c='black', label='Cluster Head')
    circle = plt.Circle((x,y),Radius, color='g', fill=False)
    plt.gcf().gca().add_artist(circle)
plt.show()

# Making list of cluster name
cluster_name=[]
dataframe_column_in = []
for you in range(Total_Cluster):
    iiicol = 'C'+str(you) + 'in'
    c='C'+str(you)
    dataframe_column_in.append(iiicol)
    cluster_name.append(c)


# Making a Data-frame with node as rows and cluster in column to see intersection and the coverage statistic
Dataframe_node_cluster= pandas.DataFrame(index=node_number_list, columns=cluster_name)
for zz in range(numNodes):
    for xx in range(len(cluster_name)):
        Dataframe_node_cluster.loc[zz, cluster_name[xx]] = data.loc[zz, dataframe_column_in[xx]]
print('Data-frame for checking node intersection')
print(Dataframe_node_cluster)
# Making CSV file
Dataframe_node_cluster.to_csv("node_cluster_data.csv", sep='\t')

# Making another Data-frame with clusters as rows and columns to see number of common nodes between clusters
Dataframe_cluster_cluster= pandas.DataFrame(index=cluster_name, columns=cluster_name)
for i in range(len(dataframe_column_in)):
    for j in range(len(dataframe_column_in)):
        s = data[dataframe_column_in[i]] & data[dataframe_column_in[j]]
        Dataframe_cluster_cluster.loc[cluster_name[i], cluster_name[j]] = sum(s)

# Adding Tranmission Power into Dataframe_cluster_cluster Dataframe
value_list = []
for key, value in Dictionary_power.items():
    value_list.append(value)
Dataframe_cluster_cluster['Power'] = value_list
#print(Dataframe_cluster_cluster)
# Making CSV file
Dataframe_cluster_cluster.to_csv("Cluster_cluster_Power_Consumption.csv", sep=',')

# Making Dataframe for Dijktra's shortest path algorithm if there is path then value would be 1 else 0.
Dataframe_Dijktra = pandas.DataFrame(index=cluster_name, columns=cluster_name)
for k in range(len(cluster_name)):
    for kk in range(len(cluster_name)):
        if Dataframe_cluster_cluster.loc[cluster_name[k], cluster_name[kk]]>0:
            Dataframe_Dijktra.loc[cluster_name[k], cluster_name[kk]] = 1
        else:
            Dataframe_Dijktra.loc[cluster_name[k], cluster_name[kk]] = 0
print('Data_frame for geting common nodes between clusters')
print(Dataframe_Dijktra)
# Making CSV file
Dataframe_Dijktra.to_csv("Cluster_cluster_connection.csv", sep=',')

# Build data stucture for graph's input
fromToDict = {'from':[], 'to':[]}
edgeList = []
for c1 in range(len(cluster_name)):
    for c2 in range(c1+1, len(cluster_name)):
        if Dataframe_cluster_cluster.loc[cluster_name[c1],cluster_name[c2]] > 0:
            fromToDict['from'].append(cluster_name[c1])
            fromToDict['to'].append(cluster_name[c2])
            fromToDict['from'].append(cluster_name[c2])
            fromToDict['to'].append(cluster_name[c1])
            edgeList.append((cluster_name[c1], cluster_name[c2], 1))
            edgeList.append((cluster_name[c2], cluster_name[c1], 1))
netData = pandas.DataFrame(fromToDict)

# Preparing Network Graph
plt.figure(3)
plt.title('Backbone Network')
#G=nx.from_pandas_dataframe(netData, 'from', 'to', create_using=nx.Graph())
G=nx.from_pandas_edgelist(netData, 'from', 'to', create_using=nx.Graph())
# Plotting backbone network
nx.draw_networkx(G, with_labels=True)
plt.show()

# cluster_name dictionary for keeping track of cluster wise power consumption
dico = {}
for fog in cluster_name:
    dico[fog] = 0

# Number of hop and power consumption calculation
hop_count_list = []

# Generating 100000 transaction for getting power consumption graph and hop count graph
Dijk_Dict = {}
for transection in range(transection_range):
    foo = node_number_list
    secure_random = random.SystemRandom()
    From_A = secure_random.choice(foo)
    To_B = secure_random.choice(foo)
    From_cluster_C = ''
    To_cluster_D = ''
    Hop = 0
    print('transaction going on: ', transection)

    for V in Dataframe_node_cluster.columns:
        check_value = Dataframe_node_cluster.loc[From_A, V]
        if check_value == True:
            From_cluster_C = V
            break

    for W in Dataframe_node_cluster.columns:
        another_check_value = Dataframe_node_cluster.loc[To_B, W]
        if another_check_value == True:
            To_cluster_D =  W
            break

    if From_cluster_C == '' or To_cluster_D == '':
        continue

    if sum(netData['from'].str.contains(From_cluster_C)) <= 0:
        continue

    if sum(netData['to'].str.contains(To_cluster_D)) <= 0:
        continue

    if From_cluster_C == To_cluster_D:
        Hop = 0
        hop_count_list.append(Hop)
        power_consumption = Dataframe_cluster_cluster.loc[From_cluster_C, 'Power']
        power_consumption = 10 ** (power_consumption / 10) / 1000  ## power consumption in watt
        dico[From_cluster_C] = dico.get(From_cluster_C, 0) + power_consumption
        continue

    if((From_cluster_C, To_cluster_D) in Dijk_Dict):
        #print('From-To combination exists', From_cluster_C, To_cluster_D)
        Dijk_output = Dijk_Dict[(From_cluster_C, To_cluster_D)]

    else:
        # applying Dijkstra algorithm between randomly selected nodes
        graph = Graph(edgeList)
        Dijk_output = graph.dijkstra(From_cluster_C, To_cluster_D)

    if len(Dijk_output) <= 1:
        continue
    else:
        Dijk_Dict[(From_cluster_C, To_cluster_D)] = Dijk_output
        h = len(Dijk_output)-1
        Hop = h
        hop_count_list.append(Hop)
        for ii in range(len(Dijk_output)-1):
            U = Dijk_output[ii]
            power_consumption = Dataframe_cluster_cluster.loc[U, 'Power']
            power_consumption = 10 ** (power_consumption / 10)/1000 ## power consumption in watt
            dico[U] = dico.get(U, 0) + power_consumption


print('Hop Count: ', hop_count_list)
print('Cluster_Wise Transmission Power: ', dico)

# Preparing clusterwise Power tansmission list form dictionary
somelist = []
for kecho in cluster_name:
    somelist.append(dico[kecho])

# Plotting no of hop count Histogram-graph
plt.figure(4)
plt.title('No of Hop Count')
plt.hist(hop_count_list, density=True, bins=10)
plt.xlabel('No of Hops Count')
plt.show()

# Plotting Cluster wise power consumption Bar-graph
plt.figure(5)
plt.title('Cluster_Wiser Power Consumption')
width = 1/1.5
plt.bar(range(len(cluster_name)), somelist, width, color="blue")
plt.xticks(range(len(cluster_name)), cluster_name)
plt.xlabel('Cluster Name')
plt.ylabel('Power Consumption in W')
plt.show()

endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('Compute times: ', starttime, endtime, totaltime)


