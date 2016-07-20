import tempfile
import os
import subprocess

def generate_vcd(src_file,toplevel,t=None):
    """ Simulate model *src_file* for time t in GHDL, and return the generated VCD. """
    def cmd(*args):
        subprocess.check_call(['ghdl']+list(args),cwd=dir)
    with tempfile.TemporaryDirectory() as dir:
        cmd('-i',src_file)  # import sources
        cmd('-m',toplevel)  # make a design
        t = ('--stop-time='+str(t),) if t is not None else ()
        cmd('-r',toplevel,'--vcd=out.vcd',*t)  # run the design
        with open('out.vcd','r') as out:
            return out.read()