import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

pos = {'router1':(100,30),'PLC1':(50,20),'PLC2':(100,20),'PLC3':(150,20),'Sensor1':(50,10),'Sensor2':(100,10),'Sensor3':(150,10)}


G.add_nodes_from(['router1','PLC1','PLC2','PLC3','Sensor1','Sensor2','Sensor3'])

G.add_edges_from([('router1','PLC1'),('router1','PLC2'),('router1','PLC3'),('PLC1','Sensor1'),('PLC1','Sensor2'),
                  ('PLC1','Sensor3'),('PLC2','Sensor1'),('PLC2','Sensor2'),('PLC2','Sensor3'),('PLC3','Sensor1'),
                  ('PLC3','Sensor2'),('PLC3','Sensor3')])

print(G)
nx.draw(G, pos, with_labels=True)

plt.show()
