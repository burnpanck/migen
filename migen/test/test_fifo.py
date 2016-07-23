import unittest
from itertools import count

from migen import *
from migen.genlib.fifo import SyncFIFO

from migen.test.support import SimCase


class SyncFIFOCase(SimCase, unittest.TestCase):
    class TestBench(Module):
        def __init__(self):
            self.submodules.dut = SyncFIFO(64, 2)

            self.sync += [
                If(self.dut.we & self.dut.writable,
                    self.dut.din[:32].eq(self.dut.din[:32] + 1),
                    self.dut.din[32:].eq(self.dut.din[32:] + 2)
                )
            ]

    def test_run_sequence(self):
        seq = list(range(20))
        def gen():
            for cycle in count():
                # fire re and we at "random"
                yield self.tb.dut.we.eq(cycle % 2 == 0)
                yield self.tb.dut.re.eq(cycle % 3 == 0)
                # the output if valid must be correct
                if (yield self.tb.dut.readable) and (yield self.tb.dut.re):
                    try:
                        i = seq.pop(0)
                    except IndexError:
                        break
                    self.assertEqual((yield self.tb.dut.dout[:32]), i)
                    self.assertEqual((yield self.tb.dut.dout[32:]), i*2)
                yield
        self.run_with(gen())

    def test_replace(self):
        seq = [x for x in range(20) if x % 5]
        def gen():
            for cycle in count():
                yield self.tb.dut.we.eq(cycle % 2 == 0)
                yield self.tb.dut.re.eq(cycle % 7 == 0)
                yield self.tb.dut.replace.eq(
                    (yield self.tb.dut.din[:32]) % 5 == 1)
                if (yield self.tb.dut.readable) and (yield self.tb.dut.re):
                    try:
                        i = seq.pop(0)
                    except IndexError:
                        break
                    self.assertEqual((yield self.tb.dut.dout[:32]), i)
                    self.assertEqual((yield self.tb.dut.dout[32:]), i*2)
                yield
        self.run_with(gen())

    class TestBenchWithStimulus(TestBench):
        def __init__(self):
            super(SyncFIFOCase.TestBenchWithStimulus,self).__init__()
            dut = self.dut

            c2 = Signal(max=2)
            c5 = Signal(max=5)
            c7 = Signal(max=7)
            self.sync += If(c2,c2.eq(c2-1)).Else(c2.eq(2-1))
            self.sync += If(c5,c5.eq(c5-1)).Else(c5.eq(5-1))
            self.sync += If(c7,c7.eq(c7-1)).Else(c7.eq(7-1))
            self.sync += dut.we.eq(c2 == 0)
            self.sync += dut.re.eq(c7 == 0)
            self.sync += dut.replace.eq(c5 == 1)

    def test_conversion(self):
        tb = self.TestBenchWithStimulus
        self.convert_run_and_compare(tb,cycles=2*7*5 * 2)
