import sys
# sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2023\AMD64\python_cst_libraries")
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2025\AMD64\python_cst_libraries")
import cst
import cst.results as cstr
import cst.interface as csti
import os
import numpy as np
import difflib
from settings import*


class CSTInterface:
    def __init__(self, fname):
        self.full_path = os.getcwd() + f"\{fname}"
        self.opencst()

    def opencst(self):
        print("CST opening...")
        allpids = csti.running_design_environments()
        open = False
        for pid in allpids:
            self.de = csti.DesignEnvironment.connect(pid)
            print(f"Opening {self.full_path}...")
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")
            break
        if not open:
            print("File path not found in current design environment...")
            print("Opening new design environment...")
            self.de = csti.DesignEnvironment.new()
            # self.de.set_quiet_mode(True) # suppress message box
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")

    def read(self, result_item):
        results = cstr.ProjectFile(self.full_path, True) #bool: allow interactive
        # try:
        res = results.get_3d().get_result_item(result_item)
        # except:
        #     print("No result item.")
        #     available_files = results.get_3d().get_tree_items()
        #     closest_match = difflib.get_close_matches(result_item, available_files, n=1, cutoff=0.5)
        #     if closest_match: 
        #         result_item = closest_match[0] 
        #         print(f"Fetch '{result_item}' instead.")
        #     else: result_item = None
        #     res = results.get_3d().get_result_item(result_item)
        res = res.get_data()
        return res

    def save(self):
        self.prj.modeler.full_history_rebuild() 
        #update history, might discard changes if not added to history list
        self.prj.save()

    def close(self):
        self.de.close()

    def excute_vba(self,  command):
        command = "\n".join(command)
        vba = self.prj.schematic
        res = vba.execute_vba_code(command)
        return res

    def create_para(self,  para_name, para_value): #create or change are the same
        command = ['Sub Main', 'StoreDoubleParameter("%s", "%.4f")' % (para_name, para_value),
                'RebuildOnParametricChange(False, True)', 'End Sub']
        res = self.excute_vba (command)
        return command

    def set_frequency_solver(self, fmin=1, fmax=3):
        command = ['Sub Main', 'ChangeSolverType "HF Frequency Domain"', 
                   f'Solver.FrequencyRange "{fmin}", "{fmax}"', 'End Sub']
        self.excute_vba(command)
        print("Frequency solver set")

    def set_time_solver(self, fmin=1, fmax=3):
        command = ['ChangeSolverType "HF Time Domain"', 
                   f'Solver.FrequencyRange "{fmin}", "{fmax}"']
        command = "\n".join(command)
        self.prj.modeler.add_to_history("time_solver_and_freq_range",command)
        self.save()
        print("Time solver set")

    def start_simulate(self, plane_wave_excitation=False):
        print("Solving...")
        try: # problems occur with extreme conditions
            if plane_wave_excitation:
                command = ['Sub Main', 'With Solver', 
                '.StimulationPort "Plane wave"', 'End With', 'End Sub']
                self.excute_vba(command)
                print("Plane wave excitation = True")
            # one actually should not do try-except otherwise severe bug may NOT be detected
            model = self.prj.modeler
            model.run_solver()
        except Exception as e: pass
        print("Solved")
    
    def delete_results(self):
        command = ['Sub Main',
        'DeleteResults', 'End Sub']
        res = self.excute_vba(command)
        return res

#----------------- Draw --------------------

    def create_shape(self, index, xmin, xmax, ymin, ymax, zmin, zmax): #create or change are the same
        command = ['With Brick', '.Reset ', f'.Name "solid{index}" ', 
                   '.Component "component2" ', f'.Material "material{index}" ', 
                   f'.Xrange "{xmin}", "{xmax}" ', f'.Yrange "{ymin}", "{ymax}" ', 
                   f'.Zrange "{zmin}", "{zmax}" ', '.Create', 'End With']
        return command
        # command = "\n".join(command)
        # self.prj.modeler.add_to_history(f"solid{index}",command)

    def delete_block(self, index):
        command = ['Sub Main', f'Solid.Delete "component2:solid{index}"', 'End Sub']
        res = self.excute_vba(command)
        return res

    def set_domain(self): 
        print("Setting domain...")
        sequence = np.zeros(NX*NY)
        print(f"{NX*NY} pixels in total...")
        # Define materials first
        self.update_distribution(sequence)
        command = []
        # Define shape and index based on materials
        for index, value in enumerate(sequence): 
            xi = index%NX
            yi = index//NX
            xmin = xi*DD + XX
            xmax = xmin + DD
            ymin = yi*DD + YY
            ymax = ymin + DD
            zmin = ZZ
            zmax = ZZ + HC
            command += self.create_shape(index, xmin, xmax, ymin, ymax, zmin, zmax)
        command = "\n".join(command)
        self.prj.modeler.add_to_history("domain",command)
        self.save()
        print("Domain set")

    def delete_domain(self):
        command = ['Sub Main',
        'Component.Delete "component2"', 'End Sub']
        res = self.excute_vba(command)
        return res

    def update_distribution(self, binary_sequence):
        print("Material distribution updating...")
        command_material = []
        for index, value in enumerate(binary_sequence):
            if value == 1: command_material += self.create_copper_material(index) #self.create_PEC_material(index)
            elif value == 0: command_material += self.create_air_material(index)
            else: print("undefined material")
        command_material = "\n".join(command_material)
        self.prj.modeler.add_to_history("material update",command_material)
        print("Material distribution updated")

    def create_copper_material(self, index): # create and change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                   '.Folder ""', '.Rho "8930"', '.ThermalType "Normal"', 
                   '.ThermalConductivity "401"', '.SpecificHeat "390", "J/K/kg"', 
                   '.DynamicViscosity "0"', '.UseEmissivity "True"', '.Emissivity "0"', 
                   '.MetabolicRate "0.0"', '.VoxelConvection "0.0"', 
                   '.BloodFlow "0"', '.MechanicsType "Isotropic"', 
                   '.YoungsModulus "120"', '.PoissonsRatio "0.33"', 
                   '.ThermalExpansionRate "17"', '.IntrinsicCarrierDensity "0"', 
                   '.FrqType "all"', f'.Type "Lossy metal"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Mu "1"', f'.Sigma "{58000000}"', 
                   '.LossyMetalSIRoughness "0.0"', '.ReferenceCoordSystem "Global"', 
                   '.CoordSystemType "Cartesian"', '.NLAnisotropy "False"', 
                   '.NLAStackingFactor "1"', '.NLADirectionX "1"', '.NLADirectionY "0"', 
                   '.NLADirectionZ "0"', '.Colour "0", "1", "1" ', '.Wireframe "False" ', 
                   '.Reflection "False" ', '.Allowoutline "True" ', 
                   '.Transparentoutline "False" ', '.Transparency "0" ', 
                   '.Create', 'End With']
        return command
       
    def create_PEC_material(self, index): # create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                   '.FrqType "all"', '.Type "PEC"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"',  
                   '.Create', 'End With']
        return command
    
    def create_air_material(self, index): # create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"',  
                   '.FrqType "all"', '.Type "Normal"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Epsilon "1"', '.Mu "1"', '.Create', 'End With']
        return command