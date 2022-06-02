import numpy as np
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from .help_functions import XYZLogger


class ModelPart(ForcePart):
    def __init__(self, system, model, log_name = 'md.xyz', nprint = 1):
        ForcePart.__init__(self, 'ml_ff', system)
        self.system = system
        self.model = model
        self.step = 0
        self.nprint = nprint
        self.log_name = log_name
        if not self.log_name is None:
            self.logger = XYZLogger(log_name)
            

    def _internal_compute(self, gpos, vtens): 
        numbers = self.system.numbers
        positions = self.system.pos / angstrom
        
        if self.system.cell.nvec == 0:
            rvec = np.eye(3) * 100
        else:
            rvec = self.system.cell.rvecs / angstrom
            
        list_of_properties = ['energy']
        if not gpos is None:
            list_of_properties.append('forces')
        if not vtens is None:
            list_of_properties.append('vtens')
        output = self.model.compute(positions, numbers, rvec = rvec, list_of_properties = list_of_properties)
        
        if not vtens is None:
            vtens[:, :] = output['vtens'] * electronvolt
        else:
            output['vtens'] = None
        if not gpos is None:
            gpos[:, :] = -output['forces'] / angstrom * electronvolt
        else:
            output['forces'] = None
         
        self.step += 1
        if not self.log_name is None:
            if self.step % self.nprint == 0:
                self.logger.write(numbers, positions, energy = output['energy'], rvec = rvec, vtens = output['vtens'],
                                  forces = output['forces'])
        
        return output['energy'] * electronvolt
        

def Optimize(model, positions, numbers, rvec = np.eye(3) * 100, log = None, fullcell = False): 
    system = System(numbers, positions * angstrom, rvecs = rvec.astype(np.float) * angstrom)
    
    ff = ForceField(system, [ModelPart(system, model, log_name = log, nprint = 1)])
    if fullcell:
        opt = QNOptimizer(FullCellDOF(ff, gpos_rms = 1e-07, grvecs_rms=1e-07))
    else:
        opt = QNOptimizer(CartesianDOF(ff, gpos_rms = 1e-07))
    try:
        opt.run()
    except RuntimeError as error:
        print(str(error))
    
    opt_positions = ff.system.pos / angstrom
    opt_rvec = ff.system.cell.rvecs / angstrom
    return opt_positions, opt_rvec
    
    
def NVE(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'md', screenprint = 1000):
    ff = ForceField(system, [ModelPart(system, model, log_name = name + '.xyz', nprint = nprint)])
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    
    
def NVT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'md', screenprint = 1000):
    ff = ForceField(system, [ModelPart(system, model, log_name = name + '.xyz', nprint = nprint)])
    thermo = NHCThermostat(temp = temp)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    

def NPT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal):
    ff = ForceField(system, [ModelPart(system, model, log_name = name + '.xyz', nprint = nprint)])
    thermo = NHCThermostat(temp = temp)
    baro = MTKBarostat(ff, temp = temp, press = pressure)
    tbc = TBCombination(thermo, baro)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    
    
def NVsigmaT(system, model, steps, nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal):
    ff = ForceField(system, [ModelPart(system, model, log_name = name + '.xyz', nprint = nprint)])
    thermo = NHCThermostat(temp = temp)
    baro = MTKBarostat(ff, temp = temp, press = pressure, vol_constraint = True)
    tbc = TBCombination(thermo, baro)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
