
import torch as ch

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)


class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = ch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)
