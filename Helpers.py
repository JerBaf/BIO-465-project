import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.hodgkin_huxley import HH
import neurodynex3.tools.input_factory as input_factory



def simulate_regular_spiking_HH_neuron(input_current, simulation_time, init_dict={"vm":0,"m":0.05,"h":0.6,"n":0.32}):

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]
    """

    # neuron parameters
    El = -70 * b2.mV
    EK = -90 * b2.mV
    ENa = 50 * b2.mV
    gl = 0.1*b2.msiemens
    gK = 5 * b2.msiemens
    gNa = 50 * b2.msiemens
    C = 1 * b2.ufarad

    # forming HH model with differential equations
    eqs = """
    I_e = input_current(t,i) : amp
    I_leak =  gl*(vm-El) : amp
    I_Na =  gNa*m**3*h*(vm-ENa) : amp
    I_K = gK*n**4*(vm-EK) : amp
    membrane_Im = I_e - I_Na - I_K - I_leak : amp
    alphah = .128*exp(-(vm/mV+43.)/18.)/ms : Hz
    alpham = -.32*(vm+47*mV)/(exp(-.25*(vm/mV+47))-1)/mV/ms : Hz
    alphan = -.032*(vm+45*mV)/(exp(-.2*(vm/mV+45))-1)/mV/ms : Hz
    betah = 4./(exp(-.2*(vm/mV+20))+1)/ms : Hz
    betam = .28*(vm+20*mV)/(exp(.2*(vm/mV+20))-1)/mV/ms : Hz
    betan = .5*exp(-(vm/mV+50)/40)/ms : Hz
    h_inf = alphah/(alphah + betah) : 1
    m_inf = alpham/(alpham + betam) : 1
    n_inf = alphan/(alphan + betan) : 1
    tau_h = 1/(alphah + betah) : second
    tau_m = 1/(alpham + betam) : second
    tau_n = 1/(alphan + betan) : second
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dvm/dt = membrane_Im/C : volt
    """

    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization

    neuron.vm = init_dict["vm"]
    neuron.m = init_dict["m"]
    neuron.h = init_dict["h"]
    neuron.n = init_dict["n"]
   
    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e","I_leak", "I_Na", "I_K", "m", "n", "h", 
                                      "h_inf", "m_inf", "n_inf", "tau_h", "tau_m", "tau_n"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon

def simulate_adaptive_HH_neuron(input_current, simulation_time, init_dict={"vm":0,"m":0.05,"h":0.6,"n":0.32,"p":0}):

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]
    """

    # neuron parameters
    El = -70 * b2.mV
    EK = -90 * b2.mV
    ENa = 50 * b2.mV
    gl = 0.1*b2.msiemens
    gM = 0.07 * b2.msiemens
    gK = 5 * b2.msiemens
    gNa = 50 * b2.msiemens
    C = 1 * b2.ufarad

    # forming HH model with differential equations
    eqs = """
    I_e = input_current(t,i) : amp
    I_leak =  gl*(vm-El) : amp
    I_Na =  gNa*m**3*h*(vm-ENa) : amp
    I_M = gM*p*(vm-EK) : amp
    I_K = gK*n**4*(vm-EK) : amp
    membrane_Im = I_e - I_Na - I_M - I_K - I_leak : amp
    alphah = .128*exp(-(vm/mV+43)/18)/ms : Hz
    alpham = -.32*(vm+47*mV)/(exp(-.25*(vm/mV+47))-1)/mV/ms : Hz
    alphan = -.032*(vm+45*mV)/(exp(-.2*(vm/mV+45))-1)/mV/ms : Hz
    betah = 4./(exp(-.2*(vm/mV+20))+1)/ms : Hz
    betam = .28*(vm+20*mV)/(exp(.2*(vm/mV+20))-1)/mV/ms : Hz
    betan = .5*exp(-(vm/mV+50)/40)/ms : Hz
    p_inf = 1./(exp(-.1*(vm/mV+40))+1) : 1
    h_inf = alphah/(alphah + betah) : 1
    m_inf = alpham/(alpham + betam) : 1
    n_inf = alphan/(alphan + betan) : 1
    tau_h = 1/(alphah + betah) : second
    tau_m = 1/(alpham + betam) : second
    tau_n = 1/(alphan + betan) : second
    tau_p = 2000.*ms/(3.3*exp((vm/mV + 20)/20)+exp(-(vm/mV+20)/20)) : second
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dp/dt = (p_inf-p)/tau_p : 1
    dvm/dt = membrane_Im/C : volt
    """

    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization
    neuron.vm = init_dict["vm"]
    neuron.m = init_dict["m"]
    neuron.h = init_dict["h"]
    neuron.n = init_dict["n"]
    neuron.p = init_dict["p"]

    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e","I_leak", "I_Na", "I_K", "I_M", "n", "h", 
                                      "m", "p", "h_inf", "m_inf", "n_inf", "tau_h", "tau_m",
                                      "tau_n","tau_p", "p_inf"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon

def plot_data(state_monitor, title=""):
    """Plots the state_monitor variables ["vm", "I_e", "m", "n", "h"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    """
    states = state_monitor.get_states()
    plot_nb = 5
    if "p" in states.keys():
        plot_nb += 1
    fig, axs = plt.subplots(plot_nb,1,figsize=(16,4*plot_nb))
    i = 0

    axs[i].plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
    axs[i].axis(xmin=0,
        xmax=np.max(state_monitor.t / b2.ms))
    axs[i].set_xlabel("t [ms]")
    axs[i].set_ylabel("v [mV]")
    axs[i].set_title("Output current")
    axs[i].grid()
    i+=1

    axs[i].plot(state_monitor.t / b2.ms, state_monitor.m[0]**3 / b2.volt, "black", lw=2)
    axs[i].plot(state_monitor.t / b2.ms, state_monitor.n[0]**4 / b2.volt, "blue", lw=2)
    axs[i].plot(state_monitor.t / b2.ms, state_monitor.h[0] / b2.volt, "red", lw=2)
    axs[i].set_xlabel("t (ms)")
    axs[i].set_ylabel("act./inact.")
    axs[i].legend(("m", "n", "h"))
    axs[i].axis((0,
        np.max(state_monitor.t / b2.ms),
             0,1))
    axs[i].set_title("Gating variables for standard model.")
    axs[i].grid()
    i+=1
    
    if "p" in states.keys():
        axs[i].plot(state_monitor.t / b2.ms, state_monitor.p[0] / b2.volt, "green", lw=2)
        axs[i].legend(("p"))
        axs[i].set_xlabel("t (ms)")
        axs[i].set_ylabel("act./inact.")
        axs[i].axis((0,
            np.max(state_monitor.t / b2.ms),
                0,1))
        axs[i].set_title("Gating variable for the adaptive current.")
        axs[i].grid()
        i+=1

    axs[i].plot(state_monitor.t / b2.ms, state_monitor.I_Na[0] / b2.uamp, lw=2)
    axs[i].plot(state_monitor.t / b2.ms, state_monitor.I_K[0] / b2.uamp, lw=2)
    axs[i].axis((
        0,
        np.max(state_monitor.t / b2.ms),
        min(min(state_monitor.I_Na[0] / b2.uamp), min(state_monitor.I_K[0] / b2.uamp)) * 1.1,
        max(max(state_monitor.I_Na[0] / b2.uamp), max(state_monitor.I_K[0] / b2.uamp)) * 1.1
    ))
    axs[i].legend(("INa","IK"))
    axs[i].set_xlabel("t [ms]")
    axs[i].set_ylabel("Currents [micro A]")
    axs[i].set_title("Ion currents")
    axs[i].grid()
    i+=1
     
    axs[i].plot(state_monitor.t / b2.ms, state_monitor.I_leak[0] / b2.uamp, lw=2)
    if  "I_M" in states.keys():
        axs[i].plot(state_monitor.t / b2.ms, state_monitor.I_M[0] / b2.uamp, lw=2)
        axs[i].axis((
            0,
            np.max(state_monitor.t / b2.ms),
            min(min(state_monitor.I_M[0] / b2.uamp), min(state_monitor.I_leak[0] / b2.uamp)) * 1.1,
            max(max(state_monitor.I_M[0] / b2.uamp), max(state_monitor.I_leak[0] / b2.uamp)) * 1.1
        ))
        axs[i].legend(("Ileak","IM"))
        axs[i].set_xlabel("t [ms]")
        axs[i].set_ylabel("Currents [micro A]")
        axs[i].set_title("Leak and Ion current.")
        axs[i].grid()
    else: 
        axs[i].axis((
            0,
            np.max(state_monitor.t / b2.ms),
            min(state_monitor.I_leak[0] / b2.uamp) * 1.1,
            max(state_monitor.I_leak[0] / b2.uamp) * 1.1
        ))
        axs[i].legend(("Ileak"))
        axs[i].set_xlabel("t [ms]")
        axs[i].set_ylabel("Currents [micro A]")
        axs[i].set_title("Leak current")
        axs[i].grid()
    i+=1
    
    axs[i].plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
    axs[i].axis((
        0,
        np.max(state_monitor.t / b2.ms),
        min(0,min(state_monitor.I_e[0] / b2.uamp)) * 1.1,
        max(state_monitor.I_e[0] / b2.uamp) * 1.1
    ))
    axs[i].legend("Iext")
    axs[i].set_xlabel("t [ms]")
    axs[i].set_ylabel("I [micro A]")
    axs[i].set_title("Input current")
    axs[i].grid()


    fig.suptitle(title,fontsize="x-large")
    fig.tight_layout()
    plt.show()

def extract_spike_timings(state_monitor):
    states = state_monitor.get_states()
    spike_timings = []
    t_states = states["t"]
    look_peak = True
    for i, m in enumerate(states["m"]):
        if look_peak and m > 0.999:
            spike_timings.append(t_states[i])
            look_peak = False
        if not look_peak and m < 0.5:
            look_peak = True
    return spike_timings

def plot_adaptive_behaviour(state_monitor):
    spike_timings = extract_spike_timings(state_monitor)
    adaptive_behaviour =[]
    j = 1
    next_spike = spike_timings[j]
    prev_spike = spike_timings[j-1]
    for t in state_monitor.get_states()["t"]:
        if t > next_spike and j < len(spike_timings)-1:
            j+=1
            next_spike = spike_timings[j]
            prev_spike = spike_timings[j-1]
        adaptive_behaviour.append(next_spike-prev_spike)
    fig,ax = plt.subplots(figsize=(16,9))
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("Inter-spike time [ms]")
    ax.set_title("Adaptive Behaviour",fontsize="x-large")
    ax.plot(adaptive_behaviour)
    plt.show()

def plot_three_adaptive_behaviours(state_monitors):
    fig,ax = plt.subplots(figsize=(16,9))
    for state_monitor in state_monitors:
        spike_timings = extract_spike_timings(state_monitor)
        adaptive_behaviour =[]
        j = 1
        next_spike = spike_timings[j]
        prev_spike = spike_timings[j-1]
        for t in state_monitor.get_states()["t"]:
            if t > next_spike and j < len(spike_timings)-1:
                j+=1
                next_spike = spike_timings[j]
                prev_spike = spike_timings[j-1]
            adaptive_behaviour.append(next_spike-prev_spike)
    legend = tuple(["Simulation_"+str(i) for i in range(len(state_monitors))])
    ax.legend(legend)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("Inter-spike time [ms]")
    ax.set_title("Adaptive Behaviours",fontsize="x-large")
    ax.plot(adaptive_behaviour)
    plt.show()