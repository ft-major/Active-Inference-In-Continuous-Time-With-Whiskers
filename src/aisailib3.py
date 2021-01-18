#%% md
# ## Functions
#%%
# Modules
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)


#%% md
# ## Classes
#%%
# Generative process class
class GenProc:
    def __init__(self):

        # Generative process parameters

        # Generative process s variance
        self.SigmaGP_s = 0.1
        # Two dimensional array storing angle and angle velocity initialized with his initial conditions
        self.x = np.array([1., 0.])
        # Harmonic oscillator angular frequency square (omega^2)
        self.omega2 = 0.5
        # Costant that quantify the amount of energy (friction?) that the agent can insert in the system
        self.u = 0.5
        # Array storing respectively proprioceptive sensory input (initialized with the real value x_0) and touch sensory input
        self.s = np.array([1, 0.])
        # Platform position (when is present) with respect to x_0 variable
        self.platform_position = 0.5
        # Time interval in which the platform appears
        self.platform_interval = [15, 75]


    # Step of generative process dynamic
    def dynamic(self, dt, t, action):
        self.x[0] += self.x[1]*dt
        self.x[1] += -self.omega2*self.x[0]*dt - self.u*np.tanh(action)*dt#*self.x2

    # Funciton that create agent's sensory input (two dimensional array)
    def genS(self, t):
        # Platform Action
        if t > self.platform_interval[0] and t < self.platform_interval[1]:
            if self.x[0] > self.platform_position:
                self.s[1] = 1.
                self.s[0] = self.platform_position
            else:
                self.s[0] = self.x[0]
                self.s[1] = 0.
        else:
            self.s[0] = self.x[0]
            self.s[1] = 0.
        self.s[0] += self.Sigma_s*rng.randn()
        return self.s

#%%
# Generative model class
class GenMod:
    def __init__(self):

        # Generative process parameters

        # Harmonic oscillator angular frequency (omega^2). We're assuming is equal to the one of the GP
        self.omega2 = 0.5
        # Vector \vec{\mu}={\mu_0, \mu_1} initialized with the GP initial conditions
        self.mu = np.array([1., 0.])
        # Vector \dot{\vec{\mu}}={\dot{\mu_0}, \dot{\mu_1}} inizialized with the right ones
        self.dmu = np.array([0., -self.omega2])
        # Internal variables precisions
        self.Sigma_mu = np.array([0.01, 0.01])
        # Array storing respectively proprioceptive sensory input (initialized with the real value x_0) and touch sensory input
        self.s = np.array([1, 0.])
        # Variances (inverse of precisions) of sensory input (the first one proprioceptive and the second one touch)
        self.Sigma_s = np.array([0.01, 10000])
        # Action variable
        self.a = 0
        # Costant that quantify the amount of energy (friction?) that the agent can insert in the system. In this case we're assuming is the same as the one of the GP
        self.u = 0.5
        # Gradient descent inference parameter
        self.k_mu = 0.1
        # Gradient descent action parameter
        self.k_a = 0.1

    # Touch function
    def g_touch(self, x, v, prec=10):
        return sech(prec*v)*(0.5*tanh(prec*x)+0.5)

    # Derivative of the touch function with respect to \mu_0
    def dg_dv(self, x, v, prec=10):
        return -prec*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5)

    # Derivative of the touch function with respect to \mu_2
    def dg_dx(self, x, v, prec=10):
        return sech(prec*v)*prec*0.5*(sech(prec*x))**2

    # Function that implement the update of internal variables.
    def update(self, sensory_states):
        # sensory_states argument (two dimensional array) come from GP and store proprioceptive
        # and somatosensory perception
        # Returns action increment

        self.s = sensory_states

        self.PE_mu = np.array([
            self.dmu[0]-self.mu[1],
            self.dmu[1]+self.omega2*self.mu[0]
        ])
        self.PE_s = np.array([
            self.s[0]-self.mu[0],
            self.s[1]-self.g_touch(x=self.mu[0], v=self.mu[1])  # v=self.dmu[0]?
        ])

        self.dF_dmu = np.array([
            self.omega2*self.PE_mu[1]/self.Sigma_mu[1] - self.PE_s[0]/self.Sigma_s[0] \
                -self.dg_dx(x=self.mu[0], v=self.mu[1])*self.PE_s[1]/self.Sigma_s[1],
            -self.PE_mu[0]/self.Sigma_mu[0] - self.dg_dv(x=self.mu[0], v=self.mu[1])*self.PE_s[1]/self.Sigma_s[1]
        ])

        self.dF_d_dmu = np.array([
            self.PE_mu[0]/self.Sigma_mu[0],
            self.PE_mu[1]/self.Sigma_mu[1]
        ])

        dF_da = (self.mu[0]-self.mu[1])/(self.omega2+1)*self.PE_s[0]/self.Sigma_s[0] + \
        (self.dg_dmu0(x=self.mu[2], v=(self.nu*self.mu[0]-self.mu[2]), dv_dmu0=self.mu[0])) * self.PE_s[1]/self.Sigma_s[1]

        # Learning internal parameter nu
        dF_dnu = -self.mu[0]*self.PE_mu[2]/self.Sigma_mu[2] \
        -(self.dg_dmu0(x=self.mu[2], v=(self.nu*self.mu[0]-self.mu[2]), dv_dmu0=self.mu[0])) * self.PE_s[1]/self.Sigma_s[1]

        # Internal variables update
        self.mu += self.dt*(self.dmu - eta*self.dF_dmu)
        self.dmu += -self.dt*eta_d*self.dF_d_dmu

        self.da = -self.dt*eta_a*dF_da
        #self.nu += -self.dt*eta_nu*dF_dnu



    def update(self, dt, s):
        epsilon_s = s - self.mu[0]
        epsilon_mu2 = self.mu[2] + self.omega2*self.mu[0]
        dFdmu0 = ( - self.omega2*epsilon_mu2/self.Sigma_mu2 - epsilon_s/self.Sigma_s)
        dFdmu1 = 0
        dFdmu2 = ( epsilon_mu2/self.Sigma_mu2 )
        dFds = (epsilon_s/self.Sigma_s)
        dsda = (self.mu[1]**2*self.u*np.cosh(self.a)**(-2)/self.mu[2])
        dFda = ( dFds*dsda )
        self.mu[0] += dt*self.mu[1] - self.k_mu*dFdmu0
        self.mu[1] += dt*self.mu[2] - self.k_mu*dFdmu1
        self.mu[2] += -self.k_mu*dFdmu2
        self.a += -self.k_a*dFda

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    dt=0.05
    gp = GenProc(x=[1,0], rng=rng)
    gm = GenMod(mu=[1,0,-1], rng=rng)
    data = []
    data.append([gp.x[0], gp.genS(), gm.a, np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), gm.mu[0]])
    stime = 5000
    action=0
    for t in range(stime):
        if t>stime/2 and t<stime*2/3:
            action= np.sign(gm.mu[1])
        else:
            action = 0
        gp.dynamic(dt=dt, action=action)
        s=gp.genS()
        gm.update(dt=dt, s=s)
        data.append([gp.x[0], gp.genS(), action, np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), gm.mu[0]])
    data=np.vstack(data)
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:, 1], c="black", lw=1)
    plt.plot(data[:, 2], c="red", lw=3)
    plt.plot(data[:, 3])
    plt.subplot(212)
    plt.plot(data[:, 4])

    # %%
    s, gpx, gmmu, gpA, gmA = gp.genS(), gp.x[0], gm.mu[0], np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), np.sqrt(gm.mu[0]**2 + (gm.mu[1]/gm.omega2)**2)
    data.append([s, gpx, gmmu, gpA, gmA])
    print("t=",t, "x0=",gp.x[0], "s=", s, "a=",gm.a)
    for t in range(10):#stime):
        #if t > 30000:
        #    gp.x[0] = np.minimum(0.5, gp.x[0])
        gp.dynamic(dt=0.0005, action=gm.a)
        s, gpx, gmmu, gpA, gmA = gp.genS(), gp.x[0], gm.mu[0], np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), np.sqrt(gm.mu[0]**2 + (gm.mu[1]/gm.omega2)**2)
        gm.update(dt=0.0005, s=gp.genS())
        print("t=",t, "x0=",gp.x[0], "s=", s, "a=",gm.a)
        data.append([s, gpx, gmmu, gpA, gmA])
    data = np.vstack(data)
    # %%
    data = []
    stime = 200000
    s, gpx, gmmu, gpA, gmA = gp.genS(), gp.x[0], gm.mu[0], np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), np.sqrt(gm.mu[0]**2 + (gm.mu[1]/gm.omega2)**2)
    data.append([s, gpx, gmmu, gpA, gmA])
    print("t=",t, "x0=",gp.x[0], "s=", s, "a=",gm.a)
    for t in range(10):#stime):
        #if t > 30000:
        #    gp.x[0] = np.minimum(0.5, gp.x[0])
        gp.dynamic(dt=0.05, action=gm.a)
        s, gpx, gmmu, gpA, gmA = gp.genS(), gp.x[0], gm.mu[0], np.sqrt(gp.x[0]**2 + (gp.x[1]/gp.omega2)**2), np.sqrt(gm.mu[0]**2 + (gm.mu[1]/gm.omega2)**2)
        gm.update(dt=0.0005, s=gp.genS())
        print("t=",t, "x0=",gp.x[0], "s=", s, "a=",gm.a)
        data.append([s, gpx, gmmu, gpA, gmA])
    data = np.vstack(data)

    # %%

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:, 1], c="red", lw=1, ls="dashed")
    plt.plot(data[:, 3], c="#aa6666", lw=3)
    plt.subplot(212)
    plt.plot(data[:, 2], c="green", lw=1, ls="dashed")
    plt.plot(data[:, 4], c="#66aa66", lw=3)
    plt.show()
