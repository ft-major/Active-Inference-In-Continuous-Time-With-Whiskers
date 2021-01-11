import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)

def sech(x):
    return 2/(np.exp(x)+np.exp(-x))

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def f(x, a, h):
    return a - h*x + np.pi/2


def p2std(p):
    return 10000*np.exp(-p)

#p2std(9)
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

    def __init__(self, dt, omega2_GP=0.5, alpha=1):

        self.omega2 = omega2_GP                                         # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.a = alpha                                                  # Harmonic oscillator amplitude (no really)
        self.x = np.array([1.,0., self.a/(self.omega2 + 1)])            # Vector x={x_0, x_1, x_2} initialized with his initial conditions
        self.s = np.array([self.a/(self.omega2 + 1), 0.])               # Array storing respectively proprioceptive sensory input (initialized with the real
                                                                        # value x_2) and touch sensory input
        self.Sigma_s = 1.                                               # Variance of the Gaussian noise that gives proprioceptive sensory input
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
# g_1(x) = \text{sech}(10x) = \frac{ 2 }{ e^{10 x} + e^{-10 x} } \\
# g_2(x) = \frac{ 1 }{ 2 } \text{tanh}(10x) +\frac{ 1 }{ 2 } = \frac{ 1 }{ 2 } \left( \frac{ e^{10x}-e^{-10x} }{ e^{10x}+e^{-10x}  } + 1 \right)
# $$
# for later is important to notice that
# $$
# \frac{ d g_1 }{ dx } = -10 \text{sech}(10x) \text{tanh}(10x)\\
# \frac{ d g_2 }{ dx } = 5 \text{sech}^2(10x)
#$$
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
# ## Gradients
# $$
# \begin{align}
# \frac{ \partial F }{ \partial \mu_0 } &= \omega^2 \frac{ \varepsilon_{\mu_1} }{ \Sigma_{\mu_1} } - \nu \frac{ \varepsilon_{\mu_2} }{ \Sigma_{\mu_2} } - \frac{ \partial g(\vec{\mu}, \nu) }{ \partial \mu_0 } \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\\
# \frac{ \partial F }{ \partial \mu_1 } &= -\frac{ \varepsilon_{\mu_0} }{ \Sigma_{\mu_0} } \\
# \frac{ \partial F }{ \partial \mu_2 } &= \frac{ \varepsilon_{\mu_2} }{ \Sigma_{\mu_2} } - \frac{ \varepsilon_{s_0} }{ \Sigma_{s_0} } - \frac{ \partial g(\vec{\mu}, \nu) }{ \partial \mu_2 } \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\\
# \frac{ \partial F }{ \partial \dot{\mu}_0 } &= \frac{ \varepsilon_{\mu_0} }{ \Sigma_{\mu_0} }\\
# \frac{ \partial F }{ \partial \dot{\mu}_1 } &= \frac{ \varepsilon_{\mu_1} }{ \Sigma_{\mu_1} }\\
# \frac{ \partial F }{ \partial \dot{\mu}_2 } &= \frac{ \varepsilon_{\mu_2} }{ \Sigma_{\mu_2} }
# \end{align}
# $$
# ## Internal variables dynamics
# $$
# \left[ \begin{matrix} \mu_0(t+dt) \\ \mu_1(t+dt) \\ \mu_2(t+dt) \end{matrix} \right] =
#	\left[ \begin{matrix} \mu_0(t) + dt \, (\dot{\mu}_0(t) - \eta \frac{ \partial F }{ \partial \mu_0 }) \\
#						  \mu_1(t) + dt \, (\dot{\mu}_1(t) - \eta \frac{ \partial F }{ \partial \mu_1 }) \\
#						  \mu_2(t) + dt \, (\dot{\mu}_2(t) - \eta \frac{ \partial F }{ \partial \mu_2 })
#	\end{matrix} \right]
# $$
# $$
# \left[ \begin{matrix} \dot{\mu}_0(t+dt) \\ \dot{\mu}_1(t+dt) \\ \dot{\mu}_2(t+dt) \end{matrix} \right] =
#	\left[ \begin{matrix} \dot{\mu}_0(t) - dt \, \eta_d \frac{ \partial F }{ \partial \dot{\mu}_0 } \\
#						  \dot{\mu}_1(t) - dt \, \eta_d \frac{ \partial F }{ \partial \dot{\mu}_1 } \\
#						  \dot{\mu}_2(t) - dt \, \eta_d \frac{ \partial F }{ \partial \dot{\mu}_2 }
#	\end{matrix} \right]
# $$
# with $\eta$ and $\eta_d$ gradient descent parameters respectively of $\vec{\mu}$ and $\dot{\vec{\mu}}$
# ## Action
# the agent modifies a certain variable of the GP (in our case the alpha parameter) by a quantity given by
# $$
# a = dt \eta_a \left( \frac{ \partial F }{ \partial s_0 }\frac{ \partial s_0 }{ \partial \alpha  } + \frac{ \partial F }{ \partial s_1 }\frac{ \partial s_1 }{ \partial \alpha  } \right)
#       = dt \eta_a \left( \frac{ \varepsilon_{s_0} }{ \Sigma_{s_0} }\frac{ \partial s_0 }{ \partial \alpha  } + \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\frac{ \partial s_1 }{ \partial \alpha  } \right)
# $$
# with
# $$
# \begin{align}
# \frac{ \partial s_0 }{ \partial \alpha } &= \frac{ \cos (\omega t) + \omega \sin (\omega t) }{ \omega^2 + 1 } = \frac{ x_0 - x_1}{ \omega^2 + 1 }  \approx ? \frac{ \mu_0 - \mu_1 }{ \omega^2 + 1 }\\
# \frac{ \partial s_1 }{ \partial \alpha } &= ? = -10 \, x_0 \, \text{sech}\left(10 (\alpha x_0 - x_2)\right) \, \text{tanh}\left(10 (\alpha x_0 - x_2)\right) \, \left( \frac{ 1 }{ 2 }\text{tanh}(10 x_2) + \frac{ 1 }{ 2 } \right) \approx ? -10 \, \mu_0 \, \text{sech}\left(10 (\alpha \mu_0 - \mu_2)\right) \, \text{tanh}\left(10 (\alpha \mu_0 - \mu_2)\right) \, \left( \frac{ 1 }{ 2 }\text{tanh}(10 \mu_2) + \frac{ 1 }{ 2 } \right)
# \end{align}
# $$
# ## Learning of the $\nu$ parameter
# $$
# \nu(t+dt) = \nu (t) - dt \eta_{\nu} \frac{ \partial F }{ \partial \nu }
# $$
# with
# $$
# \frac{ \partial F }{ \partial \nu } = - \mu_0 \frac{ \varepsilon_{\mu_2} }{ \Sigma_{mu_2} } - \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} }\frac{ \partial g(\vec{\mu}, \nu) }{ \partial \nu  } =
#   - \mu_0 \frac{ \varepsilon_{\mu_2} }{ \Sigma_{mu_2} } - \frac{ \varepsilon_{s_1} }{ \Sigma_{s_1} } \left[ -10 \, \mu_0 \, \text{sech}\left(10 (\nu \mu_0 - \mu_2)\right) \, \text{tanh}\left(10 (\nu \mu_0 - \mu_2)\right) \, \left( \frac{ 1 }{ 2 }\text{tanh}(10 \mu_2) + \frac{ 1 }{ 2 } \right) \right]
# $$
#%%
class GM:

    def __init__(self, dt, eta=0.0005, eta_d=1000, eta_a=0.001, eta_nu=0.001, omega2_GM=0.5, nu=1):

        self.omega2 = omega2_GM                                         # Harmonic oscillator angular frequency
        self.nu = nu                                                    # Harmonic oscillator amplitude (no really)
        self.mu = np.array([1.,0., self.a/(self.omega2+1)])             # Vector \vec{\mu}={\mu_0, \mu_1, \mu_2} initialized with the GP initial conditions
        self.dmu = np.array([0.,-self.omega2, (self.nu*self.mu[0]-self.mu[2])])
                                                                        # Vector \dot{\vec{\mu}}={\dot{\mu_0}, \dot{\mu_1}, \dot{\mu_2}} inizialized with the right ones
        self.Sigma_s = np.array([1.,1.])                                # Variances (inverse of precisions) of sensory input (the first one proprioceptive and the second one touch)
        self.Sigma_mu = np.array([1.,1.,1.])                            # Internal variables precisions
        self.da = 0                                                     # Action variable
        self.dt = dt                                                    # Size of a simulation step
        self.eta = np.array([eta, eta_d, eta_a, eta_nu])                        # Gradient descent weights

    def g_touch(self, prec=10, x, v):                                   # Touch function
        return sech(prec*v)*(0.5*tanh(prec*x)+0.5)

    def dg_dmu0(self, prec=10, x, v, dv_dmu0):                          # Derivative of the touch function with respect to \mu_0
        return -prec*dv_dmu0*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5)

    def dg_dmu2(self, prec=10, x, v, dx_dmu2=1, dv_dmu2=-1):            # Derivative of the touch function with respect to \mu_2
        return -prec*dv_dmu2*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5) + sech(prec*v)*5*dx_dmu2*(sech(prec*x))**2

    def update(self, sensory_states):                                   # Function that implement the update of internal variables.
                                                                        # sensory_states argument (two dimensional array) come from GP and store proprioceptive
                                                                        # and somatosensory perception
                                                                        # Returns action increment

        self.s = sensory_states
        s = self.s
        S_mu, S_s = (self.Sigma_mu, self.Sigma_s)
        mu = self.mu_x
        dmu = self.dmu_x
        nu = self.nu
        om2 = self.omega2
        eta, eta_d, eta_a, eta_nu = (self.eta[0], self.eta[1], selft.eta[2], selft.eta[3])
        PE_mu = np.array([
            dmu[0]-mu[1],
            dmu[1]+om2*mu[0],
            dmu[2]-(nu*mu[0]-mu[2])
            ])
        PE_s = np.array([
            s[0]-mu[2],
            s[1]-self.g_touch(x=mu[2], v=(nu*mu[0]-mu[2]))          #v=dmu[2]?
        ])

        dF_dmu = np.array([
            om2*PE_mu[1]/S_mu[1] - nu*PE_mu[2]/S_mu[2] - self.dg_dmu0(x=mu[2],v=(nu*mu[0]-mu[2]), dv_dmu0=nu)*PE_s[1]/S_s[1],
            -PE_mu[0]/S_mu[0],
            PE_mu[2]/S_mu[2] - PE_s[0]/S_s[0] - self.dg_dmu2(x=mu[2], v=(nu*mu[0]-mu[2]))*PE_s[1]/S_s[1]
            ])

        dF_d_dmu = np.array([
            PE_mu[0]/S_mu[0],
            PE_mu[1]/S_mu[1],
            PE_mu[2]/S_mu[2]
            ])

        # Internal variables update
        self.mu += self.dt*(self.dmu - eta*dF_dmu)
        self.dmu += -self.dt*eta_d*dF_d_dmu

        # Action update
        dF_da = (self.mu[0]-self.mu[1])/(om2+1)*PE_s[0]/S_s[0] + ( dg_dmu0(x=self.mu[2], v=(self.nu*self.mu[0]-self.mu[2]), dv_dmu0=self.mu[0]) ) * PE_s[1]/S_s[1]
        self.da = -self.dt*eta_a*dF_da

        # Learning internal parameter nu
        dF_dnu = -self.mu[0]*PE_mu[2]/S_mu[2] - ( dg_dmu0(x=self.mu[2], v=(self.nu*self.mu[0]-self.mu[2]), dv_dmu0=self.mu[0]) ) * PE_s[1]/S_s[1]
        self.nu += -self.dt*eta_nu*dF_dnu
        # Efference copy
        #self.nu += self.dt*self.da
        
        return self.da

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
