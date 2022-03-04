# Internal-Lib imports
# Projekt-Lib imports
# External-Lib imports
from graphviz import Digraph
# global values

def choose_width(weight): 
	if abs(weight) < 1 :
		return str(abs(weight * 10))
	else:
		return '10.0'

def choose_color(weight):
	colors=['#B2182B','#2166AC']
	if weight < 0.0 :
		return colors[0]
	else:
		return colors[1]

def print_network(neuron_counts, weight_arrays, output_labels, input_count):
	
	net = Digraph(name='network')
	net.attr(ranksep='1.5', nodesep='0.25', splines='spline' )
	net.edge_attr.update(style='dashed')
	
	# Input- und Hidden-Layer anlegen
	
	for i in range(len(neuron_counts)-1):
		
		with net.subgraph(name='layer'+str(i)) as layer:

			layer.attr(color='white', rank='same')
			layer.node_attr.update(shape='rect', style='filled')
			layer.edge_attr.update(style='invis')
			
			for j in range(neuron_counts[i]):
				layer.node(str(j)+'layer'+str(i), str(weight_arrays[i].columns[j]), color = '#9A9A9A' if (i != 0 and j < len(weight_arrays[i-1])) or (i == 0 and j < input_count) else '#414141', fontcolor = '#FFFFFF')
			for j in range(neuron_counts[i]-1):
				layer.edge(str(j)+'layer'+str(i),str(j+1)+'layer'+str(i))
				


	# Output-layer anlegen
	
	with net.subgraph(name='layer'+str(len(neuron_counts)-1)) as output_layer:

		output_layer.attr(color='white', rank='same')
		output_layer.node_attr.update(shape = 'rect',style = 'filled')
		output_layer.edge_attr.update(style='invis')

		for j in range(neuron_counts[-1]):
			output_layer.node(str(j)+'layer'+str(len(neuron_counts)-1), label=output_labels[j], fixedsize ='true')
		for j in range(neuron_counts[-1]-1):
			output_layer.edge(str(j)+'layer'+str(len(neuron_counts)-1),str(j+1)+'layer'+str(len(neuron_counts)-1))
	
	
	# Kanten erstellen  
	
	for k in range(len(neuron_counts)-1):
		
		for i in range(neuron_counts[k]):
			current_array = weight_arrays[k]
			
			for j in range(neuron_counts[k + 1]):
				if (j < len(current_array.index) and i < len(current_array.columns)):
					weight = current_array.loc[current_array.index[j]][current_array.columns[i]]
					net.edge(str(i)+'layer'+str(k), str(j)+'layer'+str(k+1), color = choose_color(weight), penwidth = choose_width(weight), dir='none', 
							tooltip=  str(current_array.columns[i]) + ' -> ' + str(current_array.index[j]) + ' : ' + str(round(weight,2)))
				elif (j >= len(current_array.columns)) :
					weight = 0
					net.edge(str(i)+'layer'+str(k), str(j)+'layer'+str(k+1), color = choose_color(weight), penwidth = choose_width(weight), dir='none', 
							tooltip=  str(current_array.columns[i]) + ' -> ' + str(current_array.index[j]) + ' : ' + str(round(weight,2)))
	return net