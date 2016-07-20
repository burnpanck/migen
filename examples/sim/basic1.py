from migen import *
from migen.fhdl import verilog, vhdl

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

vhdl_testbench = """
-- counter testbench
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter_tb is
end counter_tb;

architecture test of counter_tb is
component top
  port ( count: buffer unsigned (3 downto 0); sys_clk, sys_rst: std_logic);
end component;
signal cnt: unsigned (3 downto 0);
signal rst, clk: std_logic;
begin
dut: top port map (count=>cnt, sys_clk=>clk, sys_rst=>rst);
process
begin
  clk <= '0';
  rst <= '1';
  wait for 10 ns;
  rst <= '0';
  wait for 10 ns;
  for i in 0 to 20 loop
    clk <= '1';
    wait for 10 ns;
    clk <= '0';
    wait for 10 ns;
  end loop;
  wait; -- wait forever; this will finish the simulation
end process;
end test;
"""

if __name__ == "__main__":
    dut = Counter()
    run_simulation(dut, counter_test(dut), vcd_name="basic1.vcd")
    dut = Counter()
    print(verilog.convert(dut,{dut.count}))
    dut = Counter()
    out = vhdl.convert(dut,{dut.count})
    out.main_source += vhdl_testbench
    out.write(__file__+'.vhd')
    print(out)

