import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)

def sech(x):
    return 2/(np.exp(x)+np.exp(-x))

def logistic(x):
    return 1/(1+np.exp(-x))

def f(x, a, h):
    return a - h*x + np.pi/2


def p2std(p):
    return 10000*np.exp(-p)

p2std(9)
#%% md
# # Generative Process
# $$
#   \dot{\vec{x}}(t) = f(\vec{x}(t), \alpha) =
#       \left[\begin{array}{ccc} 0 & 1 & 0 \\ -\omega^2 & 0 & 0 \\ \alpha &0 &-1 \end{array}\right] \cdot \vec{x}(t) =
#       \left[\begin{array}{c} x_1(t) \\ -\omega^2 x_0(t) \\ \alpha x_0(t) - x_2(t) \end{array}\right] \nonumber
# $$
# ## Initial conditions
# $$
# \vec{x}(0) = \left[ \begin{array}{c} x_0(0) \\ x_1(0) \\ x_2(0) \end{array} \right] = \left[ \begin{array}{c} 1 \\ 0 \\ \frac{ \alpha }{ \omega^2 +1 } \end{array} \right]
# $$
# ## Solution
# $$
# \vec{x}(t) =
#   \left[ \begin{array}{c} x_0(t) \\ x_1(t) \\ x_2(t) \end{array} \right] =
#   \left[ \begin{array}{c} \cos(\omega t) \\ - \omega \sin(\omega t) \\ \frac{ \alpha ( \cos (\omega t) + \omega \sin (\omega t) ) }{ \omega^2 + 1 } \end{array} \right]
# $$
#
# From $x_2$ is extracted the proprioceptive sensory input
# $$
# s_0 (t) = x_2(t) + \mathcal{N}(s_0;0,\Sigma_{s_0}^{GP})
# $$

#%%
class GP:                                                               # Generative Process Class

    def __init__(self, dt, omega2_GP=0.5, alpha=2):

        self.omega2 = omega2_GP                                         # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.a = alpha                                                  # Harmonic oscillator amplitude ()
        self.x = np.array([1.,0., self.a/(self.omega2 + 1)])            # Vector x={x_0, x_1, x_2} initialized with his initial conditions
        self.s = np.array([self.a/(self.omega2 + 1), 0.])               # Array storing respectively proprioceptive sensory input (initialized with the real
                                                                        # value x_2) and touch sensory input
        self.Sigma_s = 1                                                # Variance of the Gaussian noise that gives proprioceptive sensory input
        self.dt = dt                                                    # Size of a simulation step
        self.t = 0                                                      # Time variable
        self.platform_position = 0.5                                    # Platform position (when is present) with respect to x_2 variable
        self.platform_interval = [15, 75]                               # Time interval in which the platform appears


    def update(self, action):                                           # Function that implement dynamics of the process.
                                                                        # Action argument (double) is the variable that comes from the GM that modifies alpha
                                                                        # variable affecting the amplitude of the oscillation.

        self.t += dt                                                    # Increment of time variable
        self.a += self.dt*action                                        # Increment of alpha variable (that changes the amplitude) given by agent's action
        self.x[0] += self.dt*(self.x[1])                                # GP dynamics implementation
        self.x[1] += self.dt*(-self.omega2*self.x[0])
        self.x[2] += self.dt*(self.a*self.x[0] - self.x[2])
        if self.t in self.platform_interval:                            # Platform Action
            if self.x[2] > self.platform_position:
                self.s[1] = 1.
                self.x[2] = self.platform_position
            else:
                self.s[0] = 0.
        else:
            self.s[1] = 0.
        self.s[0] = self.x[2] + self.Sigma_s*rng.randn()

    def platform_for_graph(self):
        plat = []
        for t in np.arange(0.,self.t,self.dt):
            if t in self.platform_interval:
                plat.append([t, self.platform_position])
        return np.vstack(plat)



#%% md
# # Generative Model
# ## Agent beliefs
# $$
# \dot{\vec{\mu}}(t) = f_{dyn}(\vec{\mu}(t), \nu) + z_{\vec{\mu}}=
#       \left[\begin{array}{ccc} 0 & 1 & 0 \\ -\omega^2 & 0 & 0 \\ \nu & 0 & -1 \end{array} \right] \cdot \vec{\mu}(t) + \mathcal{N}(\dot{\vec{\mu}}; 0, \hat{\Sigma}_{\vec{\mu}})
# $$
# with $\hat{\Sigma}_{\vec{\mu}}$ covariance matrix of the multidimensional gaussian noise. In our case it is a diagonal matrix with diagonal $\Sigma_{\mu_0}, \Sigma_{\mu_1}, \Sigma_{\mu_2}$
# $$
# s_p(\vec{\mu}(t), \nu) =
#       \left[\begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right]^T \cdot \vec{\mu}(t) + \mathcal{N}(s_p; 0, \Sigma_{s_0})
# $$
# $$
# s_t(\vec{\mu}(t), \nu) = g(\vec{\mu}(t),\nu) + \mathcal{N}(s_t; 0, \Sigma_{s_1}) =
#       g_1 \left( \left[\begin{array}{c} \nu \\ 0 \\ -1 \end{array} \right]^T \cdot \vec{\mu}(t) \right) g_2 \left( \left[\begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right]^T \cdot \vec{\mu}(t) \right)
#       + \mathcal{N}(s_t; 0, \Sigma_{s_1})
# $$
# with
# $$
# g_1(x) = \frac{ 2 }{ \pi }\text{arctan}\left( \frac{ 1 }{ (10 x)^2} \right) \\
# g_2(x) = \frac{ 1 }{ \pi } \left( \text{arctan}(10 x)+\frac{\pi}{2} \right)
# $$
# for later is important to notice that
# $$
# \frac{ d g_1 }{ dx } = \frac{ 2 }{ \pi }\frac{ 1 }{ (10 x)^{-4} +1 }
#%% md
# ## Free Energy
# $$
# F \approx
#   \frac{1}{2} \left[ \frac{(\dot{\mu_0}-\mu_1)^2}{\Sigma_{\mu_0}}
#                       + \frac{(\dot{\mu_1}+ \omega^2 \mu_0)^2}{\Sigma_{\mu_1}}
#                       + \frac{(\dot{\mu_2}-(\nu \mu_0 - \mu_2))^2}{\Sigma_{\mu_2}}
#                       + \frac{(s_0-\mu_2)^2}{\Sigma_{s_0}}
#                       + \frac{(s_1-g(\vec{\mu}, \nu))^2}{\Sigma_{s_1}} \right]
# $$
# ## Prediction errors
# $$
# \begin{align}
# \varepsilon_{\mu_0} &= \dot{\mu_0}-\mu_1 \\
# \varepsilon_{\mu_1} &= \dot{\mu_1}+ \omega^2 \mu_0 \\
# \varepsilon_{\mu_2} &= \dot{\mu_2}-(\nu \mu_0 - \mu_2) \\
# \varepsilon_{s_0} &= s_0-\mu_2 \\
# \varepsilon_{s_1} &= s_1-g(\vec{\mu}, \nu)
# \end{align}
# $$
#%% md
# ## Gradients
# $$
# \begin{align}
# \frac{ \partial F }{ \partial \mu_0 } &= \omega^2 \frac{ \varepsilon_{\mu_1} }{ \Sigma_{\mu_1} } - \nu \frac{ \varepsilon_{\mu_2} }{ \Sigma_{\mu_2} } - \frac{ \partial g(\vec{\mu}, \nu) }{ \partial \mu_0 } \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\\
# \frac{ \partial F }{ \partial \mu_1 } &= -\frac{ \varepsilon_{\mu_0} }{ \Sigma_{\mu_0} } \\
# \frac{ \partial F }{ \partial \mu_2 } &= \frac{ \varepsilon_{\mu_2} }{ \Sigma_{\mu_2} } - \frac{ \varepsilon_{s_0} }{ \Sigma_{s_0} } - \frac{ \partial g(\vec{\mu}, \nu) }{ \partial \mu_2 } \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\\
# \end{align}
# $$
# with
# $$
# \begin{align}
# \frac{ \partial g(\vec{\mu}, \nu) }{ \partial \mu_0 } =
#%%
class GM:
    """ Generative Model.

    Attributes:
        pi_s: (float) Precision of sensory probabilities.
        pi_x: (float) Precision of hidden states probabilities.
        pi_nu: (float) Precision of hidden causes probabilities.
        h: (float) Integration step of hidden states dynamics.
        gamma: (float) Attenuation factor of sensory prediction error.
        mu_s: (float)  sensory channel (central value).
        mu_x: (float) hidden state (central value).
        dmu_x: (float) Change of  hidden state (central value).
        mu_nu: (float) Internal cause (central value).
        da: (float) Increment of action
        dt: (float) Integration step
        eta: (float) Free energy gradient step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self, dt, eta=0.0005, eta_d=1000,
                 freq=0.001, amp=np.pi/2):

        self.pi_s = np.array([9,9])
        self.pi_x = np.array([9,9,9])
        self.omega_s = p2std(self.pi_s)
        self.omega_x = p2std(self.pi_x)

        self.mu_x = np.array([1.,0.,amp*1])
        self.dmu_x = np.array([0.,-1/freq,0.])
        self.nu = amp

        self.da = 1
        self.dt = dt
        self.eta = eta
        self.eta_d = eta_d
        self.freq = freq

    def f_touch(self, x, v):
        return sech(10*np.pi*v)*logistic(10*np.pi*x)

    def d_f_touch_dmu0(self, x, v):
        return -a_touch*sech(a_touch*x)*tanh(a_touch*x)*(1/2 * tanh(a_touch*x-2) + 1/2)

    def d_f_touch_dmu1(self, x, v):
        return sech(a_touch*v)*5*(sech(a_touch*x-2))**2

    def update(self, sensory_states):
        """ Update dynamics and give action

            Args:
                sensory_states: float current real proprioceptive and
                    somstosensory perception

            Returns:
                (float) current action increment
         """

        # update sensory states and dynamic precision
        self.s = sensory_states
        self.da = self.mu_x[0]

        s = self.s
        oms, omx = (self.omega_s, self.omega_x)
        mx = self.mu_x
        dmx = self.dmu_x
        n = self.nu
        da, fr = self.da, self.freq

        # TODO: gradient descent optimizations
        self.gd_mu_x = np.array([
            -(1/omx[2])*n*(n*mx[0]-mx[2]-dmx[2]) - (1/omx[1])*(mx[0]+dmx[1]) + (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*self.d_f_touch_dmu0(mx[0],mx[1]) ,
            -(1/omx[0])*fr*(mx[1]*fr-dmx[0]) + (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*self.d_f_touch_dmu1(mx[0],mx[1]),
            (1/oms[0])*(s[0]-mx[2]) - (1/omx[2])*(dmx[2]-(n*mx[0]-mx[2]))
            ])

        self.gd_dmu_x = np.array([
            -(1/omx[0])*(dmx[0] - fr*mx[1]),
            -(1/omx[1])*(mx[0] + dmx[1]),
            -(1/omx[2])*(dmx[2] - (n*mx[0] - mx[2]))])

        self.touch = self.f_touch(mx[0],mx[1])
        self.gd_nu = -(1/omx[2])*mx[0]*(n*mx[0] - mx[2] - dmx[2])
        self.gd_a = (1/oms[0])*da*(s[0]-mx[2]) - (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*da

        # classic Active inference internal variables dynamics
        eta_mu = self.eta
        eta_dmu = self.eta_d
        d_dmu_x = self.dt*( eta_dmu*self.gd_dmu_x )
        self.mu_x = self.mu_x + self.dt*( self.dmu_x + eta_mu*self.gd_mu_x)
        self.dmu_x = self.dmu_x + d_dmu_x


        self.nu += self.dt*self.gd_a
        return self.gd_a

#%%

if __name__ == "__main__":
    dt = 0.0005
    n_steps = 200000
    gp = GP(dt=dt)
    #gm = GM(dt=0.0005, eta=0.1, freq=0.5, amp=1)

#%%
    data = []
    a = 0.
    for step in np.arange(n_steps):
        gp.update(a)
        data.append([gp.x[2], gp.a])
    data = np.vstack(data)
    platform = gp.platform_for_graph()

    plt.figure(figsize=(10, 6))
    plt.subplot(111)
    plt.plot(np.arange(0,n_steps*dt,dt), data[:, 0], c="red", lw=1, ls="dashed")
    plt.plot(np.arange(0,n_steps*dt,dt), data[:, 1], c="#aa6666", lw=3)
    plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5)

    # %%
    data = []
    platform = []
    a = 0.0
    stime = 200000
    for t in range(stime):
        touch = 0.
        if t > 30000 and t<150000:
            if gp.mu_x[2]>0.5:
                touch = 1.
                gp.mu_x[2] = 0.5
                platform.append([t,0.5])

        gp.update(a)
        s, gpm, gmm, gpa, gmn = gp.s, gp.mu_x[2], gm.mu_x[2], gp.a, gm.nu
        a = gm.update( [s,touch] )
        data.append([s, gpm, gmm, gpa, gmn])
    data = np.vstack(data)

    # %%
    platform = np.vstack(platform)
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:, 1], c="red", lw=1, ls="dashed")
    plt.plot(data[:, 3], c="#aa6666", lw=3)
    plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5)
    plt.subplot(212)
    plt.plot(data[:, 2], c="green", lw=1, ls="dashed")
    plt.plot(data[:, 4], c="#66aa66", lw=3)
    plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5)
    plt.show()
