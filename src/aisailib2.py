import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)

def sech(x):
    return 1/np.cosh(x)


def tanh(x):
    return np.tanh(x)

time = np.arange(0, 30, 0.01)
coseno = 1*np.cos(time)
peak = peak_touch(coseno)
plt.plot(time, coseno)
plt.plot(time, peak)

# %%

# Generative Process Class
class GP:

    def __init__(self, dt, omega2_GP=0.5, alpha=1):

        # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.omega2 = omega2_GP
        # Harmonic oscillator amplitude (not exaclty)
        self.a = alpha
        # Vector x={x_0, x_1, x_2} initialized with his initial conditions
        self.x = np.array([1., 0.,alpha/(omega2_GP+1)])
        # Array storing respectively proprioceptive sensory input (initialized with the real value x_2) and touch sensory input
        self.s = np.array([self.a/(self.omega2 + 1), 0.])
        # Variance of the Gaussian noise that gives proprioceptive sensory input
        self.Sigma_s = 0.01
        # Size of a simulation step
        self.dt = dt
        # Time variable
        self.t = 0
        # Platform position (when is present) with respect to x_2 variable
        self.platform_position = 0.5
        # Time interval in which the platform appears
        self.platform_interval = [15, 90]

    def touch(self, x, platform_position, prec=100):
        return 0.5 * tanh(prec*(x-platform_position) + 1)

    # Function that implement dynamics of the process.
    def update(self, action):
        # Action argument (double) is the variable that comes from the GM that modifies alpha
        # variable affecting the amplitude of the oscillation.

        # Increment of time variable
        self.t += self.dt
        # Increment of alpha variable (that changes the amplitude) given by agent's action
        self.a += action
        # GP dynamics implementation
        self.x[0] += self.dt*(self.x[1])
        self.x[1] += self.dt*(-self.omega2*self.x[0])
        self.x[2] += self.dt*(self.a*self.x[0] - self.x[2])
        # Platform Action
        if self.t > self.platform_interval[0] and self.t < self.platform_interval[1]:
            self.s[1] = self.touch(self.x[2], self.platform_position)
            if self.x[2]>self.platform_position:
                self.x[2] = self.platform_position
        else:
            self.s[1] = 0.
        self.s[0] = self.x[2] + self.Sigma_s*rng.randn()

    def platform_for_graph(self):
        plat = []
        for t in np.arange(0., self.t, self.dt):
            if t in self.platform_interval:
                plat.append([t, self.platform_position])
        return np.vstack(plat)

#%%

#Generative model class
class GM:

    def __init__(self, dt, eta=0.1, eta_d=0.1, eta_a=1, eta_nu=1, omega2_GM=0.5, nu=1):

        # Harmonic oscillator angular frequency
        self.omega2 = omega2_GM
        # Harmonic oscillator amplitude (not exactly)
        self.nu = nu
        # Vector \vec{\mu}={\mu_0, \mu_1, \mu_2} initialized with the GP initial conditions
        self.mu = self.nu/(self.omega2+1)
        # Vector \dot{\vec{\mu}}={\dot{\mu_0}, \dot{\mu_1}, \dot{\mu_2}} inizialized with the right ones
        self.dmu = (self.nu*self.mu[0]-self.mu[2])
        # Variances (inverse of precisions) of sensory input (the first one proprioceptive and the second one touch)
        self.Sigma_s = np.array([0.01, 100.1])
        # Internal variables precisions
        self.Sigma_mu = 0.01
        # Action variable (in this case the action is intended as the increment of the variable that the agent is allowed to modified)
        self.da = 0
        # Size of a simulation step
        self.dt = dt
        # Gradient descent weights
        self.eta = np.array([eta, eta_d, eta_a, eta_nu])

    # Touch function
    def g_touch(self, x, v, prec=50):
        return sech(prec*v)*(0.5*tanh(x)+0.5)

    # Derivative of the touch function with respect to v
    def dg_dv(self, x, v, prec=50):
        return -prec*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5)

    # Derivative of the touch function with respect to x
    def dg_dx(self, x, v, prec=50):
        return sech(prec*v)*0.5*(sech(x))**2


    # Function that implement the update of internal variables.
    def update(self, sensory_states, x):
        # sensory_states argument (two dimensional array) come from GP and store proprioceptive
        # and somatosensory perception
        # Returns action increment

        self.s = sensory_states
        eta, eta_d, eta_a, eta_nu = (self.eta[0], self.eta[1], self.eta[2], self.eta[3])

        self.PE_mu = self.dmu - (self.nu*x - self.mu)

        self.PE_s = np.array([
            self.s[0]-self.mu,
            self.s[1]-self.g_touch(x=self.mu, v=self.dmu)
        ])

        self.dF_dmu = self.PE_mu/self.Sigma_mu - self.PE_s[0]/self.Sigma_s[0] - dg_dx(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        self.dF_d_dmu = self.PE_mu[2]/self.Sigma_mu[2] - self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        # Action update
        # case with dg/da = 1
        self.da = -self.dt*eta_a*self.PE_s[1]/self.Sigma_s[1]
        # case with real dg/da
        #self.da = -self.dt*eta_a*x*self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        # Learning internal parameter nu
        #self.nu += -self.dt*eta_nu*(-x*self.PE_mu/self.Sigma_mu -x*self.dg_dv(x=self.mu, v=self.dmu)*self.self.PE_s[1]/self.Sigma_s[1])

        self.mu += self.dt*(self.dmu - eta*self.dF_dmu)
        self.dmu += -self.dt*eta_d*self.dF_d_dmu

        # Efference copy
        self.nu += self.da

        return self.da


# %%

if __name__ == "__main__":
    dt = 0.005
    n_steps = 20000+5000
    gp = GP(dt=dt, omega2_GP=0.5, alpha=1)
    gm = GM(dt=dt, eta=0.001, eta_d=1., eta_a=0.06, eta_nu=0.01, omega2_GM=0.5, nu=1)

    data_GP = []
    data_GM = []
    a = 0.
    for step in np.arange(n_steps):
        a = gm.update(gp.s, gp.x[0])
        gp.update(a)
        data_GP.append([gp.x[2], gp.a, gp.s[0], gp.s[1], gp.x[0]])
        data_GM.append([gm.mu, gm.nu, 0, 0, gm.PE_mu,
                        gm.PE_s[0], gm.PE_s[1], gm.dmu, 0, 0, 0, 0, gm.g_touch(x=gm.mu, v=gm.dmu) ])
    data_GP = np.vstack(data_GP)
    data_GM = np.vstack(data_GM)
    platform = gp.platform_for_graph()
    #%%

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0], c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 7], c="red", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 11], c="purple", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 12], c="grey", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), gm.g_touch(x=data_GM[:, 0], v=data_GM[:, 7]), label=r"touch")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    #plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0],
             c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0],
             c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 10]*gm.Sigma_s[1], c="orange", label=r"$\, \frac{ d\varepsilon_{s_1} }{ da }$")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    #%%
    plt.figure(figsize=(20, 10))
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 9], label=r"$\frac{ 1 }{ \Sigma_{s_0} } \, \frac{ d\varepsilon_{s_0} }{ da }$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 10], c = "orange", label=r"$\frac{ 1 }{ \Sigma_{s_1} } \, \frac{ d\varepsilon_{s_1} }{ da }$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=1, ls="dashed", label=r"$x_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=3, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5, label="platform")
    # plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 2], label="Proprioceptive sensory input")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 3], label="Touch sensory input")
    #plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5, label="platform")
    # plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 2], label=r"$PE_{\mu 0}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(312)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 3], label=r"$PE_{\mu 1}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(313)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 4], label=r"$PE_{\mu 2}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 5], label=r"$PE_{s 0}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 6], label=r"$PE_{s 1}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    #%%


    # %% Testing touch function
    gm = GM(dt=0.005)
    omega = np.sqrt(gm.omega2)
    nu = 0.5 # gm.nu
    time = np.arange(0, 30, 0.001)
    x0 = np.cos(omega*time)
    x1 = -omega*np.sin(omega*time)
    x2 = nu*(np.cos(omega*time)+omega*np.sin(omega*time))/(omega**2+1)
    dx2 = nu*(-np.sin(omega*time)+omega**2*np.cos(omega*time))/(omega**2+1)
    touch = gm.g_touch(x=x2, v=dx2)
    #dtouch_dx0 = gm.dg_dmu0(x=x2, v=dx2, dv_dmu0=nu)
    dtouch_dnu = x2*gm.dg_dv(x=x2, v=dx2) + x2*gm.dg_dx(x=x2, v=dx2)

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(time, x0, label=r"$x_0$", c='#1f77b4')
    plt.plot(time, x2, label=r"$x_2$", c='#ff7f0e')
    #plt.plot(time, touch, label=r"touch function $g$", c='#9467bd')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(time, x1, label=r"$x_1$", c='#2ca02c')
    plt.plot(time, dx2, label=r"$\frac{dx_2}{dt}$", c='#d62728')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(time, x2, label=r"$x_2$", c='#ff7f0e')
    plt.plot(time, touch, label=r"touch function $g$", c='#9467bd')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.plot(time, x0, label=r"$x_0$", c='#1f77b4')
    #plt.plot(time, dtouch_dx0, label=r"$\frac{dg}{dx_0}$", c='#8c564b')
    # plt.plot(time, dtouch_dx2, label=r"$\frac{dg}{dx_2}$", c='#7f7f7f')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.plot(time, x2, label=r"$x_2$", c='#ff7f0e')
    plt.subplot(212)
    plt.plot(time, 2*dtouch_dx2, label=r"$\frac{dg}{dx_2}$", c='#7f7f7f')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.xlim([0,4])
    plt.show()
