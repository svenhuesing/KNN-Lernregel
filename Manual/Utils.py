import threading
import time
import io

import ipywidgets as widgets

import numpy as np
from matplotlib import pyplot as plt

class ThreadedExecutor(threading.Thread):
    
    def __init__(self, method, callback = None, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.method = method
        self.callback = callback
    
    def run(self):
        self.method(*self.args, **self.kwargs)
        if (self.callback): self.callback()

def df_to_widget(df):
    return widgets.HTML(df.style.set_table_attributes('class="table"').render())

class DescriptionWidget():
    
    def __init__(self, widget, description):
        self.widget = widget
        self.description_widget = widgets.HTML('<h5>' + str(description) + '</h5>')
        self.graphics = widgets.HBox([self.description_widget, widget])
    
    def description(self, description):
        self.description_widget.value = '<h5>' + str(description) + '</h5>'

class AnimatedHeatmap():
    
    def __init__(self, animation_method, width = 10, height = 10, animation_check = None):
        self.data = np.array([[0.0 for y in range(height)] for x in range(width)])
        self.animation_method = animation_method
        self.animation_check = animation_check
        
        self.image = widgets.Image(
            value = io.BytesIO().getvalue(),
            format = 'png',
            width = 400,
            height = 400
        )
        
        self.active = True
        
        ThreadedExecutor(self.animate).start()
    
    def show(self, target_data):
        display(widgets.HBox([widgets.HTML('<h2> Aktuell </h2>'), widgets.HTML('<h1> HeatMaps </h1>'), widgets.HTML('<h2> Ziel </h2>')], layout = widgets.Layout(justify_content = 'space-around', width = '100%')))
        scale = widgets.Image(
            value = open("./Manual/images/skala.png", "rb").read(),
            format = 'png',
            width = 82,
            height = 348,
        )
        buf = io.BytesIO()
        plt.imsave(buf, arr=target_data, cmap='RdBu', vmin=-1.0, vmax=1.0, format='png', origin='lower')
        target = widgets.Image(
            value = buf.getvalue(),
            format = 'png',
            width = 400,
            height = 400,
        )
        buf.close()
        display(widgets.HBox([self.image, scale, target]), layout = widgets.Layout(justify_content = 'space-around', width = '100%'))
    
    def animate(self):
        while(self.active):
            if (self.animation_check != None and self.animation_check()):
                self.animation_method(self.data)
                self.render()
            time.sleep(1.001)
    
    def render(self):
        buf = io.BytesIO()
        plt.imsave(buf, arr=self.data, cmap='RdBu', vmin=-1.0, vmax=1.0, format='png', origin='lower')
        self.image.value = buf.getvalue()
        buf.close()