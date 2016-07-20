from io import StringIO
import subprocess

from migen import *
from migen.fhdl import verilog, vhdl
from migen.sim.ghdl import generate_vcd

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
        vcd = generate_vcd([__file__+'.vhd'],'top_testbench',t='200ns')
        print(vcd)
