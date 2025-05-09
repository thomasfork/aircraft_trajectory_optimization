''' integrators for simulating dynamics models'''
import casadi as ca

def idas_integrator(zmx, umx, zdot, dt):
    ''' ode simulator based on SUNDIALS IDAS '''
    prob    = {'x':zmx, 'p':umx, 'ode':zdot}
    try:
        # try integrator setup for casadi >= 3.6.0
        znewint = ca.integrator('zint','idas',prob, 0, dt)
    except NotImplementedError:
        setup   = {'t0':0, 'tf':dt}
        znewint = ca.integrator('zint','idas',prob, setup)

    if isinstance(zmx, ca.SX):
        zmx = ca.MX('z', zmx.shape)
    if isinstance(umx, ca.SX):
        umx = ca.MX('z', umx.shape)
    znew    = znewint(x0=zmx,p=umx)['xf']
    f_znew   = ca.Function('znew',[zmx,umx],[znew],['z','u'],['znew'])
    return f_znew
