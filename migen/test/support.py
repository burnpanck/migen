import tempfile, os, subprocess, re, shutil, itertools

from migen import *
from migen.fhdl import verilog, vhdl

class SimCase:
    def setUp(self, *args, **kwargs):
        self.tb = self.TestBench(*args, **kwargs)

    def test_to_verilog(self):
        out = verilog.convert(self.tb)
        # print(out)

    def test_to_vhdl(self):
        out = vhdl.convert(self.tb)
        # print(out)

    def run_with(self, generator):
        run_simulation(self.tb, generator)

    def convert_run_and_compare(self, testbench_class, cycles, keep_files_on_fail='failed_test_files'):
        """ Convert to VHDL/Verilog and run through a simulator if available.
        Then, compare the generated VCD files. Any mismatch is considered a failure.
        """
        import numpy as np
        from io import StringIO

        from migen.sim.vcd import dump_VCD_events, VCD_events
        from migen.sim.ghdl import generate_vcd

        def n_cycles(n):
            for k in range(n):
                yield

        converter = vhdl.Converter()

        clocks = {'sys':10}
        with tempfile.TemporaryDirectory() as dir:
            # Migen simulation
            refvcdfile = os.path.join(dir,'ref.vcd')
            tb = testbench_class()
            run_simulation(tb, {n_cycles(cycles)}, vcd_name=refvcdfile,clocks=clocks)
            with open(refvcdfile, 'r') as fh:
                vcdref = dump_VCD_events(VCD_events(fh))

            # VHDL simulation
            tb = testbench_class()
            out = converter.convert(tb)
            out.main_source += converter.generate_testbench(out,clocks=clocks)
            vhdlfile = os.path.join(dir,'testbench.vhd')
            out.write(vhdlfile)
            print(out)
            try:
                ghdl_version = subprocess.check_output(['ghdl', '--version'])
            except FileNotFoundError as ex:
                self.skipTest("GHDL seems to be unavailable")
            migen_vhd = os.path.abspath(os.path.join(os.path.dirname(vhdl.__file__),'migen.vhd'))
            rawvhdlvcd = generate_vcd(
                [vhdlfile,migen_vhd], 'top_testbench',
                stoptime=str(cycles * 10) + 'ns',
                keep_files_on_fail=keep_files_on_fail,
            )
            vhdlvcd = dump_VCD_events(VCD_events(StringIO(rawvhdlvcd)))

            # compare VHDL output with reference
            vhdlvcd = {
                re.match(r'dut\.(.*?)(\[\d+:\d+])?$', k).group(1): v for k, v in vhdlvcd.items()
                if k.startswith('dut.')
            }
            for v in vhdlvcd.values():
                v._time //= 1000000
            mismatch = {}
            for k in set(vhdlvcd) & set(vcdref):
                v = vhdlvcd[k]
                vr = vcdref[k]
                # skip events at start of simulation, the simulators do not seem to agree on where signals start
                starttime = max(np.r_[v.time[v.time>0][:1],vr.time[vr.time>0][:1]])
                i = np.searchsorted(v.time,starttime)
                ir = np.searchsorted(vr.time,starttime)
                n = min(v.size-i, vr.size-ir)
                mismatch[k] = (np.flatnonzero(
                    (v.time[i:][:n] != vr.time[ir:][:n])   # GHDL uses fs as default timescale
                    | (v.value[i:][:n] != vr.value[ir:][:n])
                ),n)
            print('Matchin signals:')
            print('VHDL:      ',sorted(vhdlvcd))
            print('reference: ',sorted(vcdref))
            for k in sorted(set(vhdlvcd) & set(vcdref)):
                v = vhdlvcd[k]
                vr = vcdref[k]
                # skip events at start of simulation, the simulators do not seem to agree on where signals start
                starttime = max(np.r_[v.time[v.time>0][:1],vr.time[vr.time>0][:1]])
                i = np.searchsorted(v.time,starttime)
                ir = np.searchsorted(vr.time,starttime)
                sv = [hex(s).split('x',1)[1] for s in v.value[i:][:80]]
                svr = [hex(s).split('x',1)[1] for s in vr.value[ir:][:80]]
                w = max(len(s) for s in sv + svr)
                n = 160 // (w + 1)
                print('Signal "%s" fails on %d out of %d events' % (k, mismatch[k][0].size, mismatch[k][1]))
                print('|'.join(s.rjust(w) for s in sv[:n]))
                print('|'.join(s.rjust(w) for s in svr[:n]))
                if mismatch[k][0].size:
                    print(i,v.time[:i+2],v.value[:i+2])
                    print(ir,vr.time[:ir+2],vr.value[:ir+2])
            if any(m.size for m,n in mismatch.values()):
                failed_signals = '\n'.join(
                    '"%s" on %d out of %d events'%(k,m.size,n)
                    for k,(m,n) in mismatch.items()
                    if m.size
                )
                try:
                    shutil.copytree(dir, keep_files_on_fail)
                except FileExistsError:
                    pass
                try:
                    with open(os.path.join(keep_files_on_fail,'vhdl.vcd'),'x') as fh:
                        fh.write(rawvhdlvcd)
                except Exception as ex:
                    print('Failed to store VHDL VCD output:',ex)
                    pass
                raise AssertionError(
                    'GHDL simulation of VHDL output does not match reference simulation in Migen on the following signals:\n'
                    + failed_signals
                )


