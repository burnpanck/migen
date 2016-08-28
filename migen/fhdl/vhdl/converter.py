from functools import partial
from operator import itemgetter
import abc, itertools
import collections
import abc

from migen.fhdl.structure import *
from migen.fhdl.structure import _Operator, _Slice, _Assign, _Fragment, _Value, _Statement, _ArrayProxy
from migen.fhdl.tools import *
from migen.fhdl.visit import NodeTransformer
from migen.fhdl.visit_generic import (
    NodeTransformer as NodeTranformerGeneric,
    visitor_for, recursor_for, context_for, combiner_for,
)
from migen.fhdl.namer import build_namespace
from migen.fhdl.conv_output import ConvOutput
from migen.fhdl.bitcontainer import value_bits_sign


class Converter:
    def __init__(self,*,io_repr=all_slv,create_clock_domains=True,special_overrides={}):
        self.io_repr = io_repr
        self.create_clock_domains = create_clock_domains
        self.special_overrides = special_overrides

    def convert(self,f,ios,name='top'):
        r = ConvOutput()
        if not isinstance(f, _Fragment):
            f = f.get_fragment()
        if ios is None:
            ios = set()
        r.fragment = f
        r.name = name

        for cd_name in sorted(list_clock_domains(f)):
            try:
                f.clock_domains[cd_name]
            except KeyError:
                if self.create_clock_domains:
                    cd = ClockDomain(cd_name)
                    f.clock_domains.append(cd)
                    ios |= {cd.clk, cd.rst}
                else:
                    raise KeyError("Unresolved clock domain: '" + cd_name + "'")


        f = lower_complex_slices(f)
        insert_resets(f)
        f = lower_basics(f)
        fs, lowered_specials = lower_specials(self.special_overrides, f.specials)
        f += lower_basics(fs)

        # add explicit typing information
        f = ExplicitTyper().visit(f)

        r.lowered_fragment = f

        # generate outer structure
        ns = build_namespace(list_signals(f) \
                             | list_special_ios(f, True, True, True) \
                             | ios, _reserved_keywords)
        ns.clock_domains = f.clock_domains
        r.ns = ns

        # TODO: should the following wrapping be done in ToVHDLLowerer?
        ports = []
        replaced_signals = {}
        for io in sorted(ios, key=lambda x: x.duid):
            if io.name_override is None:
                io_name = io.backtrace[-1][0]
                if io_name:
                    io.name_override = io_name
            typ, rep = self.io_repr.VHDL_representation_for(io)
            p = Port(
                name = ns.get_name(io),
                dir = 'out' if io in outputs else 'in',
                type = typ,
                repr = rep,
            )
            replaced_signals[io] = p
            ports.append(p)
        r.ios = ios

        entity = Entity(name=name,ports=ports)
        entity_body = EntityBody(entity=entity,statements=[f])

        # convert body
        entity_body = ToVHDLLowerer(
            vhdl_repr=self.vhdl_repr,
            replaced_signals=replaced_signals,
        ).visit(entity_body)
        r.replaced_signals = replaced_signals
        r.converted = entity_body

        # generate VHDL
        src = '-- Machine generated using Migen'
        src += VHDLPrinter().visit([
            entity_body    # generates both the declaration and implementation of the entity
        ])
        r.set_main_source(
            src
        )

        return r



    def convert(self, f, ios=None, name="top",
        special_overrides=dict(),
        create_clock_domains=True,
        asic_syntax=False
    ):
        r = ConvOutput()
        if not isinstance(f, _Fragment):
            f = f.get_fragment()
        if ios is None:
            ios = set()
        r.fragment = f
        r.name = name

        for cd_name in sorted(list_clock_domains(f)):
            try:
                f.clock_domains[cd_name]
            except KeyError:
                if create_clock_domains:
                    cd = ClockDomain(cd_name)
                    f.clock_domains.append(cd)
                    ios |= {cd.clk, cd.rst}
                else:
                    raise KeyError("Unresolved clock domain: '"+cd_name+"'")

        r.ios = ios

        f = lower_complex_slices(f)
        insert_resets(f)
        f = lower_basics(f)
        fs, lowered_specials = lower_specials(special_overrides, f.specials)
        f += lower_basics(fs)

        r.lowered_fragment = f

        for io in sorted(ios, key=lambda x: x.duid):
            if io.name_override is None:
                io_name = io.backtrace[-1][0]
                if io_name:
                    io.name_override = io_name
        ns = build_namespace(list_signals(f) \
                             | list_special_ios(f, True, True, True) \
                             | ios, _reserved_keywords)
        ns.clock_domains = f.clock_domains
        r.ns = ns

        specials = self._printspecials(special_overrides, f.specials - lowered_specials, ns, r.add_data_file)


        src = "-- Machine-generated using Migen\n"
        src += self._printuse(extra=specials['use'])
        src += self._printentitydecl(f, ios, name, ns, reg_initialization=not asic_syntax)
        src += self._printarchitectureheader(f, ios, name, ns, reg_initialization=not asic_syntax, extra=specials['decl'])
        src += self._printcomb(f, ns)
        src += self._printsync(f, name, ns)
        src += specials['body']
        src += "end Migen;\n"
        r.set_main_source(src)

        return r

def generate_testbench(self, code, clocks={'sys':10}):
    """ Genertes a testbench that does nothing but instantiate the DUT and run it's clocks.

    The testbench does not generate any reset signals.

    You must supply the result of a previous conversion.
    """
    from ...sim.core import TimeManager

    name = code.name
    ns = code.ns
    f = code.lowered_fragment
    ios = code.ios

    tbname = name + '_testbench'

    sigs = list_signals(f) | list_special_ios(f, True, True, True)
    special_outs = list_special_ios(f, False, True, True)
    inouts = list_special_ios(f, False, False, True)
    targets = list_targets(f) | special_outs

    time = TimeManager(clocks)

    sortedios = sorted(ios, key=lambda x: x.duid)
    src = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.migen_helpers.all;

entity {name} is begin end {name};

architecture Migen of {name} is
component {dut}
\tport({dutport});
end component;
{signaldecl}
begin
dut: {dut} port map ({portmap});
""".format(
        name=tbname,
        dut=name,
        dutport=';\n\t\t'.join(
            self._printsig(ns, sig, 'inout' if sig in inouts else 'buffer' if sig in targets else 'in', initialise=False)
            for sig in sortedios
        ),
        signaldecl=''.join(
            'signal ' + self._printsig(ns, sig, '', initialise=True) + ';\n'
            for sig in sortedios
        ),
        portmap=', '.join(ns.get_name(s) for s in sortedios),
    )

    for k in sorted(list_clock_domains(f)):
        clk = time.clocks[k]
        src += """clk_gen({name}_clk, {dt} ns, {advance} ns, {initial});\n""".format(
            name=k,
            dt=clk.half_period,
            advance=clk.half_period - clk.time_before_trans,
            initial="'1'" if clk.high else "'0'",
        )

    src += 'end;\n'

    return src

def convert(f, ios=None, name="top", special_overrides={}, create_clock_domains=True, asic_syntax=False):
    return Converter().convert(f,ios,name,special_overrides,create_clock_domains,asic_syntax)