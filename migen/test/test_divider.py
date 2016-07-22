import unittest

from migen import *
from migen.genlib.divider import Divider
from migen.test.support import SimCase


class DivisionCase(SimCase, unittest.TestCase):
    class TestBench(Module):
        def __init__(self):
            self.submodules.dut = Divider(4)

    def test_division(self):
        def gen():
            for dividend in range(16):
                for divisor in range(1, 16):
                    with self.subTest(dividend=dividend, divisor=divisor):
                        yield self.tb.dut.dividend_i.eq(dividend)
                        yield self.tb.dut.divisor_i.eq(divisor)
                        yield self.tb.dut.start_i.eq(1)
                        yield
                        yield self.tb.dut.start_i.eq(0)
                        yield
                        while not (yield self.tb.dut.ready_o):
                            yield
                        self.assertEqual((yield self.tb.dut.quotient_o), dividend//divisor)
                        self.assertEqual((yield self.tb.dut.remainder_o), dividend%divisor)
        self.run_with(gen())

    class TestBenchWithStimulus(Module):
        def __init__(self):
            self.submodules.dut = dut = Divider(4)
            self.dividend = dividend = Signal(4)
            self.divisor = divisor = Signal(4,reset=1)
            self.sync += If(
                dut.ready_o,
                dut.start_i.eq(1),
            ).Elif(
                dut.start_i,
                dut.start_i.eq(0),
                If(
                    divisor == 15,
                    divisor.eq(1),
                    dividend.eq(dividend + 1),
                ).Else(
                    divisor.eq(divisor + 1)
                )
            )

    def test_conversion(self):
        self.convert_run_and_compare(self.TestBenchWithStimulus,cycles=15*16*5)
