import numpy as np

from yaff import *
import h5py as h5
from molmod.periodic import periodic
from .help_functions import XYZLogger

class ML_FF(ForcePart):
    def __init__(self, system, model, name = 'test.xyz', nprint = 1):
        ForcePart.__init__(self, 'ml_ff', system)
        self.system = system
        self.model = model
        self.step = 0
        self.nprint = nprint
        
        self.logger = XYZLogger(name)

    def _internal_compute(self, gpos, vtens):
        self.step += 1
        
        energy, forces, new_vtens = self.model.compute_npt(self.system.pos / angstrom, self.system.numbers, self.system.cell.rvecs / angstrom)
        
        if self.step % self.nprint == 0:
            self.logger.write(energy, self.system.cell.rvecs / angstrom, self.system.numbers, self.system.pos / angstrom, forces = forces, vtens = new_vtens)

        if not gpos is None:
            gpos[:, :] = - forces * electronvolt / angstrom

        if not vtens is None: # vtens = F x R
            vtens[:, :] = new_vtens * electronvolt
        
        return energy * electronvolt

def NVE(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', restart = None, screenprint = 1000):
    ff = ForceField(system, [ML_FF(system, model, name + '.xyz', nprint = nprint)])
    
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    #xyz = XYZWriter(name + '.xyz', start = start, step = nprint)

    #f2 = h5.File(name + '_restart.h5', mode = 'w')
    #restart_writer = RestartWriter(f2, start = start, step = 5000)

    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, hdf5_writer], temp0 = temp, restart_h5 = restart)
    verlet.run(steps)

    f.close()
    f2.close()
    
def NVT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000):
    ff = ForceField(system, [ML_FF(system, model, name + '.xyz', nprint = nprint)])

    thermo = NHCThermostat(temp = temp)

    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    #xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
    #f2 = h5.File(name + '_restart.h5', mode = 'w')
    #restart_writer = RestartWriter(f2, start = start, step = 5000)

    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, hdf5_writer], temp0 = temp)
    verlet.run(steps)

    f.close()
    #f2.close()

def NPT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal):
    ff = ForceField(system, [ML_FF(system, model, name + '.xyz', nprint = nprint)])

    thermo = NHCThermostat(temp = temp)
    
    baro = MTKBarostat(ff, temp = temp, press = pressure)
    tbc = TBCombination(thermo, baro)

    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    #xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
    #f2 = h5.File(name + '_restart.h5', mode = 'w')
    #restart_writer = RestartWriter(f2, start = start, step = 5000)

    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)

    f.close()
    #f2.close()
    
def NVsigmaT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal):
    ff = ForceField(system, [ML_FF(system, model, name + '.xyz', nprint = nprint)])

    thermo = NHCThermostat(temp = temp)
    
    baro = MTKBarostat(ff, temp = temp, press = pressure, vol_constraint = True)
    tbc = TBCombination(thermo, baro)

    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    #xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
    #f2 = h5.File(name + '_restart.h5', mode = 'w')
    #restart_writer = RestartWriter(f2, start = start, step = 5000)

    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)

    f.close()
    #f2.close()
