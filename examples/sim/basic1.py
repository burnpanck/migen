import subprocess
import re

from migen import *
from migen.fhdl import verilog, vhdl
from migen.sim.ghdl import generate_vcd
from migen.sim.vcd import dump_VCD_events, VCD_events

# Our simple counter, which increments at every cycle.
class Counter(Module):
    def __init__(self):
        self.count = Signal(4)

        # At each cycle, increase the value of the count signal.
        # We do it with convertible/synthesizable FHDL code.
        self.sync += self.count.eq(self.count + 1)


# Simply read the count signal and print it.
# The output is:
# Count: 0
# Count: 1
# Count: 2
# ...
def counter_test(dut):
    for i in range(20):
        print((yield dut.count))  # read and print
        yield  # next clock cycle
    # simulation ends with this generator


if __name__ == "__main__":
    print('*** Simulating model using Migen')
    dut = Counter()
    run_simulation(dut, counter_test(dut), vcd_name="basic1.vcd")
    print('*** Converting model to Verilog')
    dut = Counter()
    print(verilog.convert(dut,{dut.count}))
    print('*** Converting model to VHDL')
    dut = Counter()
    out = vhdl.convert(dut,{dut.count})
    out.main_source += vhdl.Converter().generate_testbench(out)
    out.write(__file__+'.vhd')
    print(out)
    print('*** Simulating model using GHDL')
    try:
        ghdl_version = subprocess.check_output(['ghdl','--version'])
    except FileNotFoundError as ex:
        print('Unable to find GHDL, cannot run VHDL simulation: ',ex)
    else:
        print('GHDL version: ',ghdl_version)
        vcd = generate_vcd([__file__+'.vhd'],'top_testbench', stoptime='210ns')
    with open('basic1.vcd','r') as fh:
        vcdref = dump_VCD_events(VCD_events(fh))
    print('*** Comparing VCD output of the two simulation runs')
    import numpy as np
    print(sorted(vcd))
    print(sorted(vcdref))

    vcd = {
        re.match(r'dut\.(.*?)(\[\d+:\d+])?$',k).group(1):v for k,v in vcd.items()
        if k.startswith('dut.')
    }
    for k in set(vcd) & set(vcdref):
        v = vcd[k]
        vr = vcdref[k]
        n = min(v.size,vr.size)
        match = np.sum(
            (v.time[:n]//1000000 == vr.time[:n])
            & (v.value[:n] == vr.value[:n])
        )
        print('Signal "%s" matches on %d out of %d events. Total events:'%(k,match,n),v.size,vr.size)

        if match < n:
            sv = [str(v*1) for v in v.value]
            svr = [str(v*1) for v in vr.value]
            n = max([len(s) for s in sv+svr])
            print('|'.join(s.rjust(n) for s in sv))
            print('|'.join(s.rjust(n) for s in svr))