# Import useful packages
import numpy as np
import time
import itertools
from collections import deque
from bokeh.plotting import figure 
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Slider, Div, Button, RadioButtonGroup
from bokeh.events import ButtonClick
from bokeh.server.server import Server
from bokeh.models import Label

kb = 1.380e-23  # Boltzmann constant
boxsize = 30e-6
partrad = 1.5e-6
run_count = 0
# Queue for holding recent particle positions
pos_x = deque('', 100)
pos_y = deque('', 100)
# List for holding time points and MSD
timepoints = []
disp = []

# Defines a class for the Langevin simulation
class Langevin2D:
    
    # Initializing variables
    def __init__(self, mden, fvisc, temp, spot):
        self.dt = 0.5e-3
        self.center = boxsize / 2
        self.partden = mden
        self.flvisc = fvisc
        self.temp = temp
        self.pot = spot * 1e-8
        self.reset()
        self.calc_params()
        
    # Reseting positions and velocities to zero
    def reset(self):
        self.x = np.zeros(2)
        self.p = np.zeros(2)
        self.pmid = np.zeros(2)
        
    # Calculating useful parameters used in solving the Langevin equation
    def calc_params(self):
        self.partmass = self.partden * 1.333 * np.pi * partrad**3
        self.stokes = 6.0 * np.pi * self.flvisc * partrad
        self.gamma = self.stokes / self.partmass
        self.c1 = np.exp(-self.gamma * self.dt)
        self.c2 = np.sqrt(kb * self.temp * (1 - self.c1**2))
        
    # Run the Langevin update
    def run(self, r1, r2):
        global pos_x, pos_y
        pos_x.append(self.x[0] * 1e6)
        pos_y.append(self.x[1] * 1e6)
        self.pmid = self.p - self.dt * self.pot * self.x / 2
        self.x = self.x + self.dt * self.pmid / 2 / self.partmass
        self.pmid = self.c1 * self.pmid + self.c2 * np.random.randn(2) * np.sqrt(self.partmass)
        self.x = self.x + self.dt * self.pmid / 2 / self.partmass
        self.p = self.pmid - self.dt * self.pot * self.x / 2
        self.check_boundary()
        self.plot_trajectories(r1, r2)
    
    # Apply periodic boundary conditions 
    def check_boundary(self):
        # Applying periodic boundaries
        for i in range(2):
            if self.x[i] <  -boxsize * 0.5: 
                self.x[i] = self.x[i] + boxsize
            if self.x[i] >=  boxsize * 0.5:
                self.x[i] = self.x[i] - boxsize
    
    # Update arrays for plotting
    def plot_trajectories(self, r1, r2):
        r1.data_source.data['x'] = pos_x
        r1.data_source.data['y'] = pos_y
        r2.data_source.data['x'] = [self.x[0] * 1e6]
        r2.data_source.data['y'] = [self.x[1] * 1e6]
    
    # Calculate MSD
    def calc_meanpos(self, timestep, r3):
        global disp, timepoints
        disp.append(np.sqrt((self.x[0] * 1e6)**2 + (self.x[1] * 1e6)**2))
        timepoints.append(self.dt * timestep) 
        r3.data_source.data['x'] = timepoints
        r3.data_source.data['y'] = disp

def modify_doc(doc):

    # Initialize figures for plotting trajectories
    p1 = figure(plot_width=500, plot_height=500, x_range=[-boxsize*1e6/2, boxsize*1e6/2], 
                y_range=[-boxsize*1e6/2, boxsize*1e6/2], title='Particle Position')
    p1.xaxis.axis_label = 'x (um)'
    p1.yaxis.axis_label = 'y (um)'
    p1.toolbar.logo = None
    p1.toolbar_location = None
    r1 = p1.circle(pos_x, pos_y, size=1, color='chocolate', alpha=1)
    r2 = p1.circle([0], [0], radius=int(partrad*1e6), alpha=0.5, color='maroon')
    
    p2 = figure(plot_width=500, plot_height=500, title='Particle Displacement', x_range=[0, 0.5],
                y_range=[0, boxsize*1e6])
    r3 = p2.circle(timepoints, disp, color='chocolate', size=1)
    p2.xaxis.axis_label = 'time (s)'
    p2.yaxis.axis_label = 'displacement (um)'
    p2.toolbar.logo = None
    p2.toolbar_location = None

 
    # Setup widgets
    TextDisp = Div(text='''<b>Note:</b> Wait for simulation  to stop before pressing buttons.''')
    StartButton = Button(label='Start', button_type="success")
    clearbutton = Button(label='Clear', button_type='warning')
    mass_density_of_particle = Slider(title='Particle mass density (kg/m^3)', value=1000, start=100, end=5000, step=100)
    fluid_viscosity = Slider(title='Dynamic viscosity (mPa.s)', value=1, start=0.01, end=1, step=0.05)
    temperature = Slider(title='Temperature (K)', value=300, start=50, end=400, step=50)
    strength_of_potential = Slider(title='Strength of quadratic potential', value=0, start=0, end=1, step=0.1)
    radiotext = Div(text='''<b>Simulation Duration</b>''')
    simulation_duration = RadioButtonGroup(labels=['Short', 'Regular', 'Long'], active=0)
    texttitle = Div(text='''<b>BROWNIAN MOTION: SOLVING THE LANGEVIN EQUATION</b>''', width=1000)
    textdesc = Div(text='''This app simulates Brownian motion of a single microscopic particle in a fluid by solving the 
                   Langevin equation. You can change the particle mass density, the fluid viscosity, and the temperature 
                   to see how it affects Brownian motion. You can also add a parabolic potential at the center and vary 
                   its strength to see how a trapped particle undergoes constrained Brownian motion''', width=1000)
    textrel = Div(text='''Learn more about how this app works and Brownian motion in <b>ES/AM 115</b>''', width=1000)
   
    # Start the Langevin simulation
    def start_Langevin_sim(event):
        # Get widget values
        mden = mass_density_of_particle.value
        fvisc = fluid_viscosity.value * 1e-3
        temp = temperature.value
        spot = strength_of_potential.value
        if simulation_duration.active == 0:
            runtime = 0.05
        elif simulation_duration.active == 1:
            runtime = 0.2
        else:
            runtime = 0.5

        global pos_x, pos_y, run_count
        pos_x.clear()
        pos_y.clear()
        # Create langevin class object
        l2d = Langevin2D(mden, fvisc, temp, spot)
        # Langevin time-steping loop
        for t in range(int(runtime/l2d.dt)):
            l2d.run(r1, r2)
            time.sleep(0.05)
            l2d.calc_meanpos(t, r3)

    def clear_sim(event):
        # Get widget values
        mden = mass_density_of_particle.value
        fvisc = fluid_viscosity.value * 1e-3
        temp = temperature.value
        spot = strength_of_potential.value
        if simulation_duration.active == 0:
            runtime = 0.05
        elif simulation_duration.active == 1:
            runtime = 0.2
        else:
            runtime = 0.5
        global pos_x, pos_y, timepoints, disp, run_count
        pos_x.clear()
        pos_y.clear()
        timepoints.clear()
        disp.clear()
        run_count = 0

        # Create langevin class object
        l2d = Langevin2D(mden, fvisc, temp, spot)
        l2d.plot_trajectories(r1, r2)
        l2d.calc_meanpos(0, r3)
 
    # Setup callbacks
    StartButton.on_event(ButtonClick, start_Langevin_sim)
    clearbutton.on_event(ButtonClick, clear_sim)
    wbox1 = widgetbox(radiotext, simulation_duration)
    wbox2 = widgetbox(TextDisp, StartButton, clearbutton)
    doc.add_root(column(texttitle, textdesc, row(mass_density_of_particle, fluid_viscosity, wbox1), 
                 row(temperature, strength_of_potential, wbox2), row(p1, p2), textrel))

server = Server({'/': modify_doc}, num_procs=1)
server.start()
 
if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()




