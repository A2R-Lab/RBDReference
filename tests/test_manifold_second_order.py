"""Fixed-spherical retracts are not affine; small angles retain first-order terms."""
import numpy as np
import pytest
import pinocchio as pin
from RBDReference import RBDReference
from RBDReference.tests.test_spherical_joint_equivalence import CASES, _grid


@pytest.mark.parametrize('fixture,build_pin,random_q,nq,nv', CASES)
def test_spherical_second_order_retract_and_step(fixture,build_pin,random_q,nq,nv):
    ref=RBDReference(_grid(fixture)); model=build_pin()
    rng=np.random.default_rng(817);q=random_q(rng);v=rng.uniform(-.4,.4,nv)
    def fd(f,n):
        h=2e-4; cols=[]
        for k in range(n):
            e=np.eye(n)[k]*h
            cols.append((8*(f(e)-f(-e))-(f(2*e)-f(-2*e)))/(12*h))
        return np.stack(cols,axis=-1)
    for arg,flag in [('q',pin.ARG0),('v',pin.ARG1)]:
        expected=fd(lambda e:np.array(pin.dIntegrate(model,q,v+e,flag)),nv)
        np.testing.assert_allclose(ref.d2Integrate(q,v,arg,'v'),expected,atol=2e-8,rtol=2e-8)
    for scale in [0.,1e-8,1e-7,1e-5,.01,.099,.101]:
        w=scale*v; b=pin.integrate(model,q,w)
        for arg,flag in [('from',pin.ARG0),('to',pin.ARG1)]:
            np.testing.assert_allclose(ref.dDifference(q,b,arg),pin.dDifference(model,q,b,flag),atol=2e-12,rtol=2e-12)
    u=np.linspace(-.2,.3,nv)
    for scheme in ('euler','semi_implicit_euler'):
        expected=fd(lambda e:ref.plant_step_gradient(ref.integrate(q,e[:nv]),
            v+e[nv:2*nv],u+e[2*nv:],.02,scheme),3*nv).transpose(0,2,1)
        np.testing.assert_allclose(ref.plant_step_hessian(q,v,u,.02,scheme),expected,atol=2e-7,rtol=2e-7)
