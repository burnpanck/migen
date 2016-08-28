from functools import partial
from operator import itemgetter
import abc, itertools
import collections
import abc

from migen.fhdl.structure import *
from migen.fhdl.structure import (
    _Operator, _Slice, _Assign, _Fragment, _Value, _Statement, _ArrayProxy
)
from migen.fhdl.tools import *
from migen.fhdl.visit import NodeTransformer
from migen.fhdl.visit_generic import (
    NodeTransformer as NodeTranformerGeneric,
    visitor_for, recursor_for, context_for, combiner_for,
)
from migen.fhdl.namer import build_namespace
from migen.fhdl.conv_output import ConvOutput
from migen.fhdl.bitcontainer import value_bits_sign

from .type_annotator import ExplicitTyper
from .syntax import reserved_keywords
from .ast import (
    Port, Entity, EntityBody
)
from .lowerer import all_slv, natural_repr, ToVHDLLowerer
from .writer import VHDLPrinter


class Converter:
    def __init__(self,*,io_repr=all_slv,create_clock_domains=True,special_overrides={}):
        self.io_repr = io_repr
        self.create_clock_domains = create_clock_domains
        self.special_overrides = special_overrides
        self.vhdl_repr = natural_repr

    def convert(self,f,ios=None,name='top'):
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

        r.lowered_fragment = f

        # pre-gather names and signals from implicit tree
        if True:
            # TODO: replace by VHDL-aware namespace
            ns = build_namespace(list_signals(f) \
                                 | list_special_ios(f, True, True, True) \
                                 | ios, reserved_keywords)
            ns.clock_domains = f.clock_domains
            r.ns = ns

        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs

        # add explicit typing information
        annotator = ExplicitTyper()
        f = annotator.visit(f)
        r.explicit_typed = f

        # generate outer structure
        # TODO: should the outer structure instead be generated in ToVHDLLowerer?
        ports = collections.OrderedDict()
        replaced_signals = {}
        for io in sorted(ios, key=lambda s: s.duid):
            if io.name_override is None:
                io_name = io.backtrace[-1][0]
                if io_name:
                    io.name_override = io_name
            typ = annotator.visit(io).type
            rep = self.io_repr.VHDL_representation_for(typ)
            p = Port(
                name = ns.get_name(io),
                dir = 'inout' if io in inouts else 'out' if io in targets else 'in',
                type = typ,
                repr = rep,
            )
            replaced_signals[io] = p
            ports[p.name] = p
        r.ios = ios

        entity = Entity(name=name,ports=ports)
        entity_body = EntityBody(
            entity=entity,
            statements=[f]
        )

        # convert body
        entity_body = ToVHDLLowerer(
            ns = ns,
            vhdl_repr=self.vhdl_repr,
            replaced_signals=replaced_signals,
        ).visit(entity_body)
        entity_body.signals = collections.OrderedDict(
            (s.name,s) for s in (
                replaced_signals[s]
                for s in sorted(sigs, key=lambda s: s.duid)
                if s in replaced_signals   # otherwise, the signal never appeared; why was it here in the first place?
            )
        )
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
