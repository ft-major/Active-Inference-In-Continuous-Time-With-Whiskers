#%% md
# ## Functions
#%%
# Modules
import numpy as np
import matplotlib.pyplot as plt


#%% md
# ## Classes
#%%
# Generative process class
class GenProc:
    def __init__(self, rng, x):

        # Generative process parameters

        # np.random.RandomState
        self.rng = rng
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


# Generative model class
class GenMod:
    def __init__(self, rng, mu):
        #self.s = s                                  # Variable that store joint angle proprioceptive input
        self.a = 0                                  # Action variable
        self.mu = mu                                # Three dimensional array styoring brain state variables mu, mu' and mu'', corresponding rispectively to angle, angle velocity and angle accerelation internal representations.
        self.omega2 = 0.5                           # Harmonic oscillator angular frequency (omega^2). We're assuming is equal to the one of the GP
        self.u = 0.005                              # Costant that quantify the amount of energy (friction?) that the agent can insert in the system. In this case we're assuming is the same as the one of the GP

        self.Sigma_s = 1                            # Generative model s variance (in this case we're assuming the agent knows gp variace)
        self.Sigma_mu2 = 1                          # Generative model mu'' variance
        self.k_mu = 0.1                             # Gradient descent inference parameter
        self.k_a = 0.01                             # Gradient descent action parameter

    def VFE(self, s):                               # Variational Free Energy
        epsilon_s = s - self.mu[0]                  # Sensory prediction error
        epsilon_mu2 = self.mu[2] + self.omega2*self.mu[0]
                                                    # Internal variable prediction error
        return 1/2*( espilon_s**2/self.Sigma_s - espilon_mu2**2/self.Sigma_mu2 )

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