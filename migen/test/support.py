import tempfile, os, subprocess, re, shutil

from migen import *
from migen.fhdl import verilog, vhdl


class SimCase:
    def setUp(self, *args, **kwargs):
        self.tb = self.TestBench(*args, **kwargs)

    def test_to_verilog(self):
        out = verilog.convert(self.tb)
        print(out)

    def test_to_vhdl(self):
        vhdl.convert(self.tb)

    def run_with(self, generator):
        run_simulation(self.tb, generator)

    def convert_run_and_compare(self, testbench_class, cycles, keep_files_on_fail='failed_test_files'):
        """ Convert to VHDL/Verilog and run through a simulator if available.
        Then, compare the generated VCD files. Any mismatch is considered a failure.
        """
        import numpy as np

        from migen.sim.vcd import dump_VCD_events, VCD_events
        from migen.sim.ghdl import generate_vcd

        def n_cycles(n):
            for k in range(n):
                yield

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
            out = vhdl.convert(tb)
            out.main_source += vhdl.Converter().generate_testbench(out,clocks=clocks)
            vhdlfile = os.path.join(dir,'testbench.vhd')
            out.write(vhdlfile)
            print(out)
            try:
                ghdl_version = subprocess.check_output(['ghdl', '--version'])
            except FileNotFoundError as ex:
                self.skipTest("GHDL seems to be unavailable")
            else:
                vhdlvcd = generate_vcd(
                    [vhdlfile], 'top_testbench',
                    stoptime=str(cycles * 10) + 'ns',
                    keep_files_on_fail=keep_files_on_fail,
                )

            # compare VHDL output with reference
            vhdlvcd = {
                re.match(r'dut\.(.*?)(\[\d+:\d+])?$', k).group(1): v for k, v in vhdlvcd.items()
                if k.startswith('dut.')
            }
            mismatch = {}
            for k in set(vhdlvcd) & set(vcdref):
                v = vhdlvcd[k]
                vr = vcdref[k]
                n = min(v.size, vr.size)
                mismatch[k] = (np.flatnonzero(
                    (v.time[:n] // 1000000 != vr.time[:n])   # GHDL uses fs as default timescale
                    | (v.value[:n] != vr.value[:n])
                ),n)
            print('Matchin signals:')
            print('VHDL:      ',sorted(vhdlvcd))
            print('reference: ',sorted(vcdref))
            for k in sorted(set(vhdlvcd) & set(vcdref)):
                v = [hex(v).split('x',1)[1] for v in vhdlvcd[k].value[:80]]
                vr = [hex(v).split('x',1)[1] for v in vcdref[k].value[:80]]
                w = max(len(s) for s in v + vr)
                n = 160 // (w + 1)
                print('Signal "%s" fails on %d out of %d events' % (k, mismatch[k][0].size, mismatch[k][1]))
                print('|'.join(s.rjust(w) for s in v[:n]))
                print('|'.join(s.rjust(w) for s in vr[:n]))
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
                raise AssertionError(
                    'GHDL simulation of VHDL output does not match reference simulation in Migen on the following signals:\n'
                    + failed_signals
                )


