import numpy as np
import matplotlib.pyplot as plt
from mpmath import sech, tanh
rng = np.random.RandomState()


def f(x, a, h):
    return a - h*x + np.pi/2


def p2std(p):
    return 10000*np.exp(-p)

#%% md
# # Generative Process
# $$
#   \dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \alpha) =
#       \left[\begin{array}{ccc} 0 & 1 & 0 \\ -\omega^2 & 0 & 0 \\ \alpha &0 &-1 \end{array}\right] \cdot \mathbf{x}(t) =
#       \left[\begin{array}{c} x_1(t) \\ -\omega^2 x_0(t) \\ \alpha x_0(t) - x_2(t) \end{array}\right] \nonumber
# $$
# ## Initial conditions
# $$
# \mathbf{x}(0) = \left[ \begin{array}{c} x_0(0) \\ x_1(0) \\ x_2(0) \end{array} \right] = \left[ \begin{array}{c} 1 \\ 0 \\ \frac{ \alpha }{ \omega^2 +1 } \end{array} \right]
# $$
# ## Solution
# $$
# \mathbf{x}(t) =
#   \left[ \begin{array}{c} x_0(t) \\ x_1(t) \\ x_2(t) \end{array} \right] =
#   \left[ \begin{array}{c} \cos(\omega t) \\ - \omega \sin(\omega t) \\ \frac{ \alpha ( \cos (\omega t) + \omega \sin (\omega t) ) }{ \omega^2 + 1 } \end{array} \right]
# $$
#
# From $x_2$ is extracted the proprioceptive sensory input
# $$
# s_0 (t) = x_2(t) + \mathcal{N}(s_0;0,\Sigma_{s_0}^{GP})
# $$

#%%
p2std(9)
class GP:                                                   # Generative Process Class

    def __init__(self, dt=0.0005, omega2_GP=0.01, alpha=0.1):

        self.omega2 = omega2_GP                                         # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.a = alpha                                                  # Harmonic oscillator amplitude ()
        self.x = np.array([1.,0., self.a/(self.omega2 + 1)])            # Vector x={x_0, x_1, x_2} initialized with his initial conditions
        self.s = self.a/(self.omega2 + 1)                               # Proprioceptive sensory input initialized with the real value (x_2)
        self.Sigma_s = 1                                                # Variance of the Gaussian noise that gives proprioceptive sensory input
        self.dt = dt                                                    # Size of a simulation step
        self.t = 0                                                      # Time variable

    def update(self, action):
        """ Update dynamics of the process.

        Args:
            action: (float) moves the current inner state.

        """

        self.a += self.dt*action
        self.mu_x[0] += self.dt*(self.mu_x[1])
        self.mu_x[1] += self.dt*(-self.freq*self.mu_x[0])
        self.mu_x[2] += self.dt*(self.a*self.mu_x[0] - self.mu_x[2])
        self.s = self.mu_x[2] + self.omega_s*rng.randn()

        return self.s

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

    def __init__(self, dt=0.0005, eta=0.0005, eta_d=1000,
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
        return sech(a_touch*v)*(1/2 * tanh(a_touch*x-2) + 1/2)

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


if __name__ == "__main__":

    gp = GP(dt=0.0005, freq=0.5, amp=1)
    gm = GM(dt=0.0005, eta=0.1, freq=0.5, amp=1)

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
