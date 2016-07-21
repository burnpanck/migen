import tempfile
import os, shutil
import subprocess

from .vcd import dump_VCD_events, VCD_events

def generate_vcd(src_files,toplevel,t=None):
    """ Simulate model *src_file* for time t in GHDL, and return the generated VCD. """
    def cmd(*args):
        subprocess.check_call(['ghdl']+list(args),cwd=dir)
    with tempfile.TemporaryDirectory() as dir:
        try:
            for f in src_files:
                shutil.copy(f,dir)
            cmd('-i',*tuple(os.path.split(p)[-1] for p in src_files))  # import sources
            cmd('-m',toplevel)  # make a design
            t = ('--stop-time='+str(t),) if t is not None else ()
            cmd('-r',toplevel,'--vcd=out.vcd',*t)  # run the design
            with open(os.path.join(dir,'out.vcd'),'r') as out:
                return dump_VCD_events(VCD_events(out))
        except Exception as ex:
            print('Files in temporary dir ',dir)
            print('\n'.join(os.listdir(dir)))
            raise
