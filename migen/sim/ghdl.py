import tempfile
import os, shutil
import subprocess
import re

def generate_vcd(src_files, toplevel, stoptime=None, keep_files_on_fail=None):
    """ Simulate model *src_file* for time t in GHDL, and return the generated VCD. """
    def cmd(*args):
        return subprocess.check_output(
            ['ghdl']+list(args),
            cwd=dir,
            stderr=subprocess.STDOUT,
        )
    with tempfile.TemporaryDirectory() as dir:
        try:
            for f in src_files:
                shutil.copy(f,dir)
            stoptime = ('--stop-time=' + str(stoptime),) if stoptime is not None else ()
            try:
                cmd('-i',*tuple(os.path.split(p)[-1] for p in src_files))  # import sources
                cmd('-m',toplevel)  # make a design
                cmd('-r', toplevel,'--vcd=out.vcd', *stoptime)  # run the design
            except subprocess.CalledProcessError as ex:
                output = (ex.output or b'').decode('ascii',errors='replace')
                if keep_files_on_fail is not None:
                    # adjust file paths in error messages
                    for p in src_files:
                        fn = os.path.split(p)[-1]
                        output = re.sub(
                            '('+re.escape(fn)+r'):(\d+)',
                            os.path.abspath(os.path.join(keep_files_on_fail,fn)).replace('\\','\\\\')+r':\2',
                            output, flags = re.MULTILINE,
                        )
                raise RuntimeError('GHDL failed with the following output:\n'+output)
            with open(os.path.join(dir,'out.vcd'),'r') as out:
                return out.read()
        except Exception as ex:
            try:
                shutil.copytree(dir,keep_files_on_fail)
            except FileExistsError:
                pass
            raise
