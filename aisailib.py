import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState()


def f(x, a, h):
    return a - h*x + np.pi/2


# %%
class GP:
    """ Generative process.

    Implementation of the generative process :

    Attributes:
        pi_s: (float) Precision of sensory probabilities.
        pi_x: (float) Precision of hidden states probabilities.
        h: (float) Integration step of hidden states dynamics.
        gamma: (float) Attenuation factor of sensory prediction error.
        mu_s: (float)  sensory channel (central value).
        mu_x: (float) hidden state (central value).
        dmu_x: (float) Change of  hidden state (central value).
        da: (float) Increment of action
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        a: (float) action
    """

    def __init__(self, eta=0.0005, freq=0.01, decay=0.3, amp=np.pi/2):

        self.pi_s = 6
        self.pi_x = 6
        self.h = 1.0/4.0
        self.mu_x = np.ones(3)
        self.mu_s = 1
        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)
        self.eta = eta
        self.freq = freq
        self.decay = decay
        self.a = amp

    def update(self, action):
        """ Update dynamics of the process.

        Args:
            action_increment: (float) moves the current inner state.

        """

        self.a += self.eta*action

        self.mu_x += self.h*np.array([
            self.mu_x[1] - self.decay*self.freq*self.mu_x[0],
            -self.freq*self.mu_x[0],
            self.a*self.mu_x[1] - self.mu_x[2]])

        self.mu_x[2] = np.minimum(np.pi, self.mu_x[2])
        self.mu_x[2] = np.maximum(-np.pi, self.mu_x[2])

        self.s = self.mu_x[2] + self.omega_s*rng.randn()
        return self.s


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
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self,eta=0.0005, freq=0.01, decay=0.3, amp=np.pi/2):

        self.pi_s = 3
        self.pi_x = 3
        self.pi_nu = 3

        self.h = 1.0
        self.gamma = 6

        self.mu_x = -np.ones(3)
        self.dmu_x = -np.ones(3)
        self.mu_nu = amp

        self.da = 1/self.h
        self.eta = eta
        self.freq = freq
        self.decay = decay

        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)

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
        self.da = 0.01#self.mu_x[2]/self.h
        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)


        # dynamics of internal variables
        self.dmu_x = np.array([
            self.mu_x[1] - self.decay*self.freq*self.mu_x[0],
            -self.freq*self.mu_x[0],
            self.mu_nu*self.mu_x[1] - self.mu_x[2]])

        s = self.s
        oms, omx = (self.omega_s, self.omega_x)
        mx = self.mu_x
        dmx = self.dmu_x
        n = self.mu_nu
        h, da, fr, d = self.h, self.da, self.freq, self.decay

        # TODO: gradient descent optimizations
        self.gd_mu_x = np.array([
            -(1/omx)*(n*(n*mx[0] - mx[2] - dmx[2]) +
                      d*fr*(d*fr*mx[0] + dmx[1])),
            -(1/omx)*(fr*(fr*mx[1] + dmx[0])),
            (1/oms**2)*(s-mx[2]) - (1/omx)*(dmx[2] - (n*mx[0] - mx[2]))])

        self.gd_dmu_x = np.array([
            -(1/omx)*(fr*mx[1] + dmx[0]),
            -(1/omx)*(d*fr*mx[0] + dmx[1]),
            -(1/omx)*(dmx[2] - (n*mx[0] - mx[2]))])

        self.gd_mu_nu =  (1/omx)*mx[0]*(dmx[2] - (n*mx[0] - mx[2]))
        self.gd_a = -(1/oms**2)*da*(s - mx[2])

        # update with gradients
        self.dmu_x += self.eta*self.gd_dmu_x
        self.mu_x += self.eta*(self.dmu_x + self.gd_mu_x)
        self.mu_nu += self.eta*self.gd_mu_nu

        return self.gd_a


if __name__ == "__main__":

    gp = GP(decay=0.2)
    gm = GM(decay=0.2)

    # %%
    data = []
    a = 2*np.pi
    stime = 10000
    for t in range(stime):
        gp.update(a)
        s, yg, ym, aa, n = gp.s, gp.mu_x[2], gm.mu_x[2], gp.a, gm.mu_nu
        a = gm.update(s)

        data.append([s, yg, ym, aa, n])
data = np.vstack(data)
plt.figure(figsize=(10, 6))
plt.plot(data[:,[4]])
