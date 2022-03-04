# Python-Internal-Libs imports
import sys
import time
import math
import random
import os

# Project-Libs imports (from './lib')
sys.path.extend([os.path.join(os.path.dirname('.'), 'lib')])
from Manual.lib.Utils import *
from Manual.lib.PrintNetwork import print_network

# External-Libs imports
import pandas as pd
import numpy as np

import ipywidgets as widgets

def default_get_target_value(x, y):
    if (y > x):
        return 0.99
    elif (y == x):
        return 0
    else:
        return -0.99

class Szenario:
    
    def __init__(self, width = range(0, 100), height = range(0, 100), scale = 0.01, hidden_neuron_count = 1, get_target_value = default_get_target_value, enable_help = True):
        self.width = width
        self.height = height
        self.scale = scale
        self.get_target_value = get_target_value
        self.enable_help = enable_help
        self.hidden_neuron_count = hidden_neuron_count
    
    def initialize_edges_values(self, knn_manual, edge_range = (-1, 1), do_display = True):
        edge_value_range = edge_range[1] - edge_range[0]
        
        edges12 = {}
        edges23 = {}
        
        for h_neuron in range(1, 1 + self.hidden_neuron_count):
            h_name = 'h_' + str(h_neuron)
            for in_neuron in range(1, 1 + 2):
                in_name = 'in_' + str(in_neuron)
                value = round((random.random() * edge_value_range) + edge_range[0], 2)
                if (in_name in edges12.keys()):
                    edges12[in_name][h_name] = value
                else:
                    edges12[in_name] = {h_name : value}
            out_name = 'out'
            value = round((random.random() * edge_value_range) + edge_range[0], 2)
            edges23[h_name] = {out_name : value}

        knn_manual.edges12 = pd.DataFrame(edges12)
        knn_manual.edges23 = pd.DataFrame(edges23)
        knn_manual.edges_matrixs = [knn_manual.edges12, knn_manual.edges23]
        
        if (do_display):
            display(widgets.HBox([widgets.HTML(value = '<h3>Initial Edges to Hidden Layer</h3>'), widgets.HTML(value = '<h3>Initial Edges to Output Layer</h3>')],
                                 layout = widgets.Layout(justify_content = 'space-around')))
            display(widgets.HBox([df_to_widget(knn_manual.edges12), df_to_widget(knn_manual.edges23)],
                                 layout = widgets.Layout(justify_content = 'space-around')))
            display(print_network([2, self.hidden_neuron_count, 1], knn_manual.edges_matrixs, ['out'], 2))

class KNN_Manual_v1:
    
    def __init__(self, szenario = Szenario()):
        self.szenario = szenario
    
    def initialize_edges_values(self, edge_range = (-1, 1), do_display = True):
        self.szenario.initialize_edges_values(self, edge_range, do_display)
    
    def create_environment(self):
        self.sliders_collection = []
        self.slider_visuals = []
        
        self.target_data = np.array([[self.szenario.get_target_value(x, y) for y in self.szenario.height] for x in self.szenario.width])
    
    def generate_animated_sliders(self):
        self.sliders_collection = []
        self.slider_visuals = []
        for edges in self.edges_matrixs:
            sliders = []
            for column in edges.columns:
                for index in edges.index:
                    slider = widgets.FloatSlider(
                        value = edges.loc[index][column],
                        orientation = 'horizontal',
                        min = -0.5,
                        max = 0.5,
                        step = 0.001,
                        description = '[' + column + ' , ' + index + '] : ',
                        layout = widgets.Layout(width = '450px')
                    )
                    sliders.append(slider)
            self.sliders_collection.append(sliders)
            self.slider_visuals.append(widgets.HBox(sliders, layout = widgets.Layout(display = 'table-cell')))
    
    def generate_animated_heatmap(self):
        scale = self.szenario.scale
        width = self.szenario.width
        height = self.szenario.height
        edges12 = self.edges12
        edges23 = self.edges23
        def animate(data):
            for x in width:
                for y in height:
                    in_v = np.tanh(np.array([[x * scale], [y * scale]]))
                    h_v = np.tanh(np.dot(edges12, in_v))
                    out_v = np.tanh(np.dot(edges23, h_v))
                    data[x][y] = out_v
        self.heatmap = AnimatedHeatmap(animate, len(width), len(height))
    
    def display_visuals(self):
        display(widgets.HBox([widgets.HTML(value = '<h3>Gewichte zum Hidden Layer</h3>'), widgets.HTML(value = '<h3>Gewichte zum Output Layer</h3>')],
                             layout = widgets.Layout(justify_content = 'space-around')))
        display(widgets.HBox(self.slider_visuals,
                             layout = widgets.Layout(justify_content = 'space-around')))
        self.heatmap.show(self.target_data)
        #self.deviation_visual = widgets.HTML('<h2> Current Deviation: ?% </h2>')
        #display(self.deviation_visual)
    
    def start_updating_edges(self):
        self.updating_edges_alive = True
        def update_edges(knn_manual = self):
            while(knn_manual.updating_edges_alive):
                for edge_index in range(len(knn_manual.edges_matrixs)):
                    edges = knn_manual.edges_matrixs[edge_index]
                    sliders = knn_manual.sliders_collection[edge_index]
                    slider_index = 0
                    for column in edges.columns:
                        for index in edges.index:
                            edges.loc[index][column] = sliders[slider_index].value
                            slider_index = slider_index + 1
                time.sleep(0.001)
        ThreadedExecutor(update_edges).start()
    
    def stop_updating_edges(self):
        self.updating_edges_alive = False
    
    def start_change_values_edges(self):
        self.change_scale_alive = True
        def change_scale(knn_manual = self):
            max_deviation = 2 * 0.99 * len(knn_manual.szenario.width) * len(knn_manual.szenario.height)
            max_scale = 2
            while(knn_manual.change_scale_alive):
                deviation = 0.0
                for x in knn_manual.szenario.width:
                    for y in knn_manual.szenario.height:
                        deviation = deviation + np.abs(knn_manual.heatmap.data[x][y] - knn_manual.target_data[x][y])
                deviation = deviation / max_deviation
                #knn_manual.deviation_visual.value = '<h2> Current Deviation: ' + str(round(deviation, 3) * 100)[:4] + '% </h2>'
                for matrix_index in range(len(knn_manual.sliders_collection)):
                    sliders = knn_manual.sliders_collection[matrix_index]
                    for slider in sliders:
                        value = slider.value
                        v = min(abs(slider.max - value), abs(slider.min - value)) / slider.max
                        slider.max = value + max_scale * deviation
                        slider.min = value - max_scale * deviation
                time.sleep(0.01)
        ThreadedExecutor(change_scale).start()
    
    def stop_change_values_edges(self):
        self.change_scale_alive = False
    
    def display(self):
        self.generate_animated_sliders()
        self.generate_animated_heatmap()
        self.display_visuals()
    
class KNN_Manual_v2:
    
    def __init__(self, szenario = Szenario()):
        self.szenario = szenario
    
    def initialize_edges_values(self, edge_range = (-1, 1), do_display = True):
        self.szenario.initialize_edges_values(self, edge_range, do_display)
    
    def create_environment(self):
        self.inputEnv = {'x': random.random(), 'y': random.random()}
        self.inputEnv['target'] = self.szenario.get_target_value(self.inputEnv['x'], self.inputEnv['y'])

        self.edges_matrixs_old = [self.edges12.copy(), self.edges23.copy()]
        self.sliders_collection = []
        self.slider_visuals = []
        
        self.target_data = np.array([[self.szenario.get_target_value(x, y) for y in self.szenario.height] for x in self.szenario.width])
        
        self.status = 0 # 0 = idle, 1 = updating edges, 2 = next
    
    # implement visuals-generating methods
    
    def generate_input_image(self, x, y, v):
        buf = io.BytesIO()
        width = range(0, 50)
        height = range(0, 50)
        x, y = (round(x, 2), round(y, 2))
        plt.imsave(buf, 
                   arr = np.array([[v 
                        if (abs(w - (x * len(width))) <= 1 and abs(h - (y * len(height))) <= 1) else
                            0.0 for h in height] for w in width]),
                   cmap='RdBu', vmin=-1, vmax=1.0, format='png', origin='lower')
        return buf.getvalue()
    
    def generate_input_vector_visuals(self):
        self.input_image_widget = widgets.Image(
            value = self.generate_input_image(self.inputEnv['x'], self.inputEnv['y'], self.szenario.get_target_value(self.inputEnv['x'], self.inputEnv['y'])),
            format = 'png',
            width = 200,
            height = 200)

        self.input_location_text_widget = widgets.HTML('<h4>Input: [x = ' + str(self.inputEnv['x'])[:4] 
            + ' , y = ' + str(self.inputEnv['y'])[:4] + ']</h4>')

        self.input_target_text_widget = widgets.HTML('<h4>Target: ' 
            + ('-' if self.szenario.get_target_value(self.inputEnv['x'], self.inputEnv['y']) < 0 else '') 
            + str(abs(round(self.szenario.get_target_value(self.inputEnv['x'], self.inputEnv['y']), 3)))[:4] + '</h4>')

        self.input_display_widget = widgets.HBox([self.input_image_widget, self.input_location_text_widget, self.input_target_text_widget],
            layout = widgets.Layout(display = 'table-cell'))
    
    def generate_error_output_visuals(self):
        self.error_text_widget = widgets.HTML('',
            layout = widgets.Layout(width = '250px'))

        self.output_text_widget = widgets.HTML('',
            layout = widgets.Layout(width = '200px'))
    
    def generate_animated_heatmap(self):
        def check_animation():
            return False # disable continuous automatic update

        self.heatmap = AnimatedHeatmap(None, len(self.szenario.width), len(self.szenario.height), check_animation)
        self.heatmap.active = False
        self.heatmap.render()
    
    def generate_next_button_visuals(self):
        self.next_input_button = widgets.Button(description = 'Next Input')
        self.next_input_button.on_click(self.next_button)
    
    def generate_animated_sliders(self):
        for edge_index in range(2):
            edges, edges_old = self.edges_matrixs[edge_index], self.edges_matrixs_old[edge_index]
            sliders = []
            for column in edges.columns:
                for index in edges.index:
                    slider = widgets.FloatSlider(
                        value = 0,
                        orientation = 'horizontal',
                        min = -1,
                        max = 1,
                        step = 0.001,
                        info = {},
                        layout = widgets.Layout(width = '250px')
                    )
                    sliders.append(DescriptionWidget(slider,
                        column + ' -> ' + index + ' : ' 
                        + ('-' if edges.loc[index][column] < 0 else '+')
                        + str(abs(edges.loc[index][column]))[:4] + ' + '))
            self.sliders_collection.append(sliders)
            self.slider_visuals.append(widgets.HBox([s.graphics for s in sliders], layout = widgets.Layout(display = 'table-cell')))
    
    def package_visuals(self):
        # Pack Output-Error/Deviation-Next-Buttoun Visuals in a HBox
        self.action_box_widget = widgets.HBox([widgets.HBox(layout = widgets.Layout(width = '50px', height = '50px')),
            self.output_text_widget, self.error_text_widget, self.next_input_button], layout = widgets.Layout(align_items = 'center'))

        # Pack all adjustment sliders in a HBox
        self.slider_content_widget = widgets.HBox(self.slider_visuals,
            layout = widgets.Layout(justify_content = 'space-around'))

        # Pack both previously packed HBoxes in a HBox
        self.content_box_widget = widgets.HBox([self.slider_content_widget, widgets.HTML('<br/><br/>'), self.action_box_widget],
        layout = widgets.Layout(display = 'table-cell', justify_content = 'space-around'))
    
    def display_visuals(self):
        display(widgets.HBox([
                widgets.HTML(value = '<h3> Input </h3>'),
                widgets.HTML(value = '<h3> Gewichte zum Hidden Layer </h3>'),
                widgets.HTML(value = '<h3> Gewichte zum Output Layer </h3>')],
            layout = widgets.Layout(justify_content = 'space-around')))

        display(widgets.HBox([self.input_display_widget, self.content_box_widget],
            layout = widgets.Layout(justify_content = 'space-around')))

        self.heatmap.show(self.target_data)
    
    def display(self):
        self.generate_input_vector_visuals()
        self.generate_error_output_visuals()
        self.generate_animated_heatmap()
        self.generate_next_button_visuals()
        self.generate_animated_sliders()
        self.package_visuals()
        self.display_visuals()
    
    # update of sliders (animation)
    
    def start_animation_sliders(self):
        self.alive = True
        def update(self):
            while self.alive :
                # sync with next_button
                if (self.status == 0):
                    self.status = 1
                    for edge_index in range(2):
                        edges, edges_old = self.edges_matrixs[edge_index], self.edges_matrixs_old[edge_index]
                        sliders = self.sliders_collection[edge_index]
                        slider_index = 0
                        for column in edges.columns:
                            for index in edges.index:
                                slider = sliders[slider_index]
                                edges.loc[index][column] = edges_old.loc[index][column] + slider.widget.value
                                slider_index = slider_index + 1
                    self.update_error()
                    self.status = 0
                time.sleep(0.01)
        ThreadedExecutor(update, None, self).start()
     
    def stop_animation_sliders(self):
        self.alive = False
    
    # implement calculation methods

    def calc_output(self, x, y):
        in_v = np.tanh(np.array([[x], [y]]))
        h_v = np.tanh(np.dot(self.edges12, in_v))
        out_v = np.tanh(np.dot(self.edges23, h_v))
        return out_v[0][0]

    def calc_error(self, v_out, v_target):
        error = (v_out - v_target) / (0.99 * 2)
        return -error if (error < 0) else error
    
    # Generate Error/Deviation - Visuals

    def update_error(self):
        output = self.calc_output(self.inputEnv['x'], self.inputEnv['y'])
        error = self.calc_error(output, self.inputEnv['target'])
        error = str(round(error, 3) * 100)[:4]
        self.error_text_widget.value = '<h3>Error: ' + str(round(self.inputEnv['target'] - output, 2)) + ' (' + error + '%)</h3>'
        output = ('-' if output < 0 else '+') + str(abs(round(output, 3)))[:4] # string has always the same length
        self.output_text_widget.value = '<h3>Output: ' + output + '</h3>'
    
    # Create Next-Button 

    def same_algebraic_sign(self, a = int, b = int):
        if (a > 0 and b > 0):
            return True
        elif (a < 0 and b < 0):
            return True

    # Implement Next-Button function (next input vector) 
    def next_button(self, update):
        # sync with update edges
        while self.status != 0:
            time.sleep(0.001)
        self.status = 2
        
        # deactive next-button
        self.next_input_button.disabled = True
        
        # generate new input-vector
        self.inputEnv['x'], self.inputEnv['y'] = (random.random(), random.random())
        self.inputEnv['target'] = self.szenario.get_target_value(self.inputEnv['x'], self.inputEnv['y'])
        
        # update animated output-heatmap
        for x in self.szenario.width:
            for y in self.szenario.height:
                self.heatmap.data[x][y] = self.calc_output(x * self.szenario.scale, y * self.szenario.scale)
        self.heatmap.render()
        
        # update error/deviation
        error = 0
        for x in self.szenario.width:
            for y in self.szenario.height:
                error = error + self.calc_error(self.calc_output(x * self.szenario.scale, y * self.szenario.scale), self.target_data[x][y])
        
        # calcualte new maximum change of values
        change_range = 2 * np.tanh(error / (len(self.szenario.width) * len(self.szenario.height) * 2))
        
        # update adjustment sliders
        for edge_index in range(2):
            edges = self.edges_matrixs[edge_index]
            sliders = self.sliders_collection[edge_index]
            slider_index = 0
            for column in edges.columns:
                for index in edges.index:
                    # cheat help:
                    change_factor = 1.0
                    if (not self.szenario.enable_help):
                        pass
                    elif (self.same_algebraic_sign(self.inputEnv['target'], self.edges23['h_1']['out'])):
                        if (column == 'in_1'):
                            change_factor = 1.5
                        elif (column == 'in_2'):
                            change_factor = 0.3
                    elif (not self.same_algebraic_sign(self.inputEnv['target'], self.edges23['h_1']['out'])):
                        if (column == 'in_1'):
                            change_factor = 0.3
                        elif (column == 'in_2'):
                            change_factor = 1.5
                    
                    slider = sliders[slider_index]
                    slider.widget.value = 0.0
                    slider.widget.min = -change_range * change_factor
                    slider.widget.max = change_range * change_factor
                    slider.description('[' + column + ' , ' + index + '] -> '
                        + ('-' if edges.loc[index][column] < 0 else '+')
                        + str(abs(edges.loc[index][column]))[:4] + ' + ')
                    slider_index = slider_index + 1
        
        # update input-vector-image
        self.input_image_widget.value = self.generate_input_image(self.inputEnv['x'], self.inputEnv['y'], self.inputEnv['target'])
        self.input_location_text_widget.value = '<h4>Input: [x = ' + str(self.inputEnv['x'])[:4] + ' , y = ' + str(self.inputEnv['y'])[:4] + ']</h4>'
        
        # update target-visuals
        target = ('-' if self.inputEnv['target'] < 0 else '') + str(abs(round(self.inputEnv['target'], 3)))[:4]
        self.input_target_text_widget.value = '<h4>Target: ' + target + '</h4>'
        
        # update environment
        self.edges_matrixs_old[0] = self.edges12.copy()
        self.edges_matrixs_old[1] = self.edges23.copy()
        
        # reactive next-button
        self.next_input_button.disabled = False
        
        self.status = 0

def create_knn_manual_v1(szenarion = Szenario(width = range(50), height = range(50))):
    knn_manual_v1 = KNN_Manual_v1(szenarion)

    knn_manual_v1.initialize_edges_values(do_display = False)
    knn_manual_v1.create_environment()
    knn_manual_v1.display()
    knn_manual_v1.start_updating_edges()
    knn_manual_v1.start_change_values_edges()
    
    return knn_manual_v1

def create_knn_manual_v2(szenarion = Szenario()):
    knn_manual_v2 = KNN_Manual_v2(szenarion)

    knn_manual_v2.initialize_edges_values(do_display = False)
    knn_manual_v2.create_environment()
    knn_manual_v2.display()
    knn_manual_v2.start_animation_sliders()
    
    return knn_manual_v2