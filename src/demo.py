from plotter import Plotter, PredErrPlotter
from sim import Sim, a2xy
from aisailib2 import GPSimple as GP, GM
import numpy as np


sidewidth = 0.8
sideheight = 1.5
center = [0, 1]
normal_box = np.array([
    (-sidewidth, -sideheight),
    (sidewidth, -sideheight),
    (sidewidth, sideheight),
    (-sidewidth, sideheight)]) + center

sidewidth = 5
sideheight = 1.5
center = [0, 1]
large_box = np.array([
    (-sidewidth, -sideheight),
    (sidewidth, -sideheight),
    (sidewidth, sideheight),
    (-sidewidth, sideheight)]) + center

for type in ["touch"]:

    print("simulating", type, "...")

    stime = 20000

    gp = GP(dt=0.005, omega2_GP=0.6, alpha=1)
    gm = GM(dt=0.005, eta=.08, eta_d=1., eta_a=0.02, eta_nu=0.02, omega2_GM=0.6, nu=1)

    points = (normal_box if type == "normal" or
              type == "touch" else large_box)
    sim = Sim("demo_"+type, points=points)

    prederr = PredErrPlotter("prederr", type, stime)
    genProcPlot = Plotter("gen_proc_"+type, type="process",
                          wallcolor=[0.2, 0.2, 0, 0.2],
                          labels={"x": "proprioception",
                                  "nu": "action (oscil. ampl.)"},
                          color=[.5, .2, 0], stime=stime)

    genModPlot = Plotter("gen_mod_"+type, type="model",
                         wallcolor=[0, 0, 0, 0],
                         labels={"x": "proprioception prediction",
                                 "nu": "internal cause (repr. oscill. ampl.)"},
                         color=[.2, .5, 0], stime=stime)

    delta_action = 0

    sens = np.zeros(stime)
    ampl = np.zeros(stime)
    sens_model = np.zeros(stime)
    ampl_model = np.zeros(stime)
    touch = np.zeros(stime)

    frame = 0
    current_touch = 0

    START_MOVE_AHEAD = stime*(16/100)
    STOP_MOVE_AHEAD = stime*(23/100)
    START_MOVE_BEHIND = stime*(80/100)
    STOP_MOVE_BEHIND = stime*(90/100)

    for t in range(stime):

        if t < stime*(16/100):
            box_pos = np.array([0, 5])
        elif START_MOVE_AHEAD < t < STOP_MOVE_AHEAD:
            e = np.exp(-(t - START_MOVE_AHEAD)/(stime*(10/100)))
            box_pos = np.array([0, 1.3 + 3.7 * e])
        elif STOP_MOVE_AHEAD < t < START_MOVE_BEHIND:
            box_pos = np.array([0, 1.3])
        elif START_MOVE_BEHIND < t < STOP_MOVE_BEHIND:
            e = np.exp(-(t -START_MOVE_BEHIND)/(stime*(10/100)))
            box_pos = np.array([0, 1.3 + 3.7 * (1 - e)])
        else:
            box_pos = np.array([0, 5])
        sim.move_box(box_pos)

        # move and conpute collision
        collision, curr_angle_limit = sim.move_box(box_pos)

        # update process
        gp.x[2] = np.minimum(curr_angle_limit, gp.x[2])
        gp.update(delta_action)

        # get state
        sens[t] = gp.s[0]
        sens_model[t] = gm.mu[2]
        ampl[t] = gp.a
        ampl_model[t] = gm.nu

        # update model and action
        delta_action = gm.update([sens[t], 1*collision])

        touch[t] = gm.g_touch(gm.mu[2], gm.dmu[2])

        # plot
        if t % int(stime/200) == 0 or t == stime - 1:

            print(frame)
            frame += 1

            sim.set_box()
            sim.update(sens[t], sens_model[t])

            wall = curr_angle_limit if \
                t>=STOP_MOVE_AHEAD and t<=START_MOVE_BEHIND \
                else None
            prederr.update([1*collision, touch[t]], t)
            genProcPlot.update([sens[t], ampl[t], wall, 0], t)
            genModPlot.update([sens_model[t], ampl_model[t], wall, 0], t)

    np.savetxt(type+"_touch", touch)
