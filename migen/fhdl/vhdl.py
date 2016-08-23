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

_reserved_keywords = {
    "abs", "access", "after", "alias", "all", "and", "architecture", "array", "assert",
    "attribute", "begin", "block", "body", "buffer", "bus", "case", "component",
    "configuration", "constant", "disconnect", "downto", "else", "elsif", "end",
    "entity", "exit", "file", "for", "function", "generate", "generic", "group",
    "guarded", "if", "impure", "in", "inertial", "inout", "is", "label", "library",
    "linkage", "literal", "loop", "map", "mod", "nand", "new", "next", "nor", "not",
    "null", "of", "on", "open", "or", "others", "out", "package", "port", "postponed",
    "procedure", "process", "pure", "range", "record", "register", "reject", "return",
    "rol", "ror", "select", "severity", "shared", "signal", "sla", "sli", "sra", "srl",
    "subtype", "then", "to", "transport", "type", "unaffected", "units", "until"
}




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
        entity_body = ToVHDLConverter(
            vhdl_repr=self.vhdl_repr,
            replaced_signals=replaced_signals,
        ).visit(entity_body)
        r.replaced_signals = replaced_signals
        r.converted = entity_body

        # generate VHDL
        src = VHDLPrinter().visit([
            entity_body    # generates both the declaration and implementation of the entity
        ])
        r.set_main_source(
            src
        )

        return r



class _MapProxy(object):
    def __init__(self,getter):
        self._getter = getter

    def __getitem__(self,item):
        return self._getter(item)




class ConverterOld:
    def typeof(self,sig):
        """ Calculate the VHDL type of a given expression/signal/variable. """
        if len(sig)==1:
            return std_logic
        base = signed if sig.signed else unsigned
        return base[len(sig)-1:0]

    def _printliteral(self, node, type):
        assert isinstance(node, Constant), "Choices in case statements must be constants"
        cp = literal_printers.get(type.ultimate_base, None)
        if cp is None:
            raise TypeError("Don't know how to print a constant of type %s"%type)
        l = type.indextypes[0].length if isinstance(type, VHDLArray) else None
        return cp(node.value, l)

    def _printsig(self, ns, s, dir, initialise=False):
        n = ns.get_name(s) + ': ' + dir
        if len(s) > 1:
            if s.signed:
                n += " signed"
            else:
                n += " unsigned"
            n += "(" + str(len(s) - 1) + " downto 0)";
        else:
            n += " std_logic"
        if initialise:
            n += ' := ' + self._printliteral(s.reset, self.typeof(s))
        return n

    def _printexpr(self, ns, node, type=None):
        printer = VHDLExprPrinter(ns,_MapProxy(self.typeof), self._convert_expr_type)
        if type is not None and type is not True:
            return printer.visit_as_type(node,type)
        expr, etype = printer.visit(node)
        if type is True:
            return expr, etype
        return expr

    def _convert_expr_type(self, type, expr, orig_type):
        if orig_type.compatible_with(type):
            return expr
        if orig_type.castable_to(type) and type.ultimate_base.name is not None:
            return type.ultimate_base.name + '(' + expr + ')'
        converter = standard_type_conversions.get((type.ultimate_base,orig_type.ultimate_base),None)
        if converter is not None:
            return converter(type,expr,orig_type)
        if not isinstance(orig_type,VHDLArray) or not isinstance(type,VHDLArray):
            raise TypeError("Don't know how to convert type %s to %s"%(orig_type,type))
        if not orig_type.ultimate_base == type.ultimate_base:
            raise TypeError("Don't know how to convert type %s to %s"%(orig_type,type))
        # simply assume the presence of a resize function
        expr = 'resize('+expr+','+str(type.indextypes[0].length)+')'
        if not orig_type.valuetype.compatible_with(type.valuetype):
            assert orig_type.castable_to(type.valuetype)
            assert type.name is not None
            expr = type.name+'('+expr+')'
        return expr

    def _printnode(self, ns, level, node):
        if isinstance(node, _Assign):
            assignment = " <= "
            assert isinstance(node.l,(Signal,_Slice))
            left,leftt = self._printexpr(ns,node.l,type=True)
            return "\t" * level + left + assignment + self._printexpr(ns, node.r, type=leftt) + ";\n"
        elif isinstance(node, collections.Iterable):
            return "".join(self._printnode(ns, level, n) for n in node)
        elif isinstance(node, If):
            r = "\t" * level + "if " + self._printexpr(ns, node.cond, integer) + "/=0 then\n"
            r += self._printnode(ns, level + 1, node.t)
            if node.f:
                r += "\t" * level + "else\n"
                r += self._printnode(ns, level + 1, node.f)
            r += "\t" * level + "end if;\n"
            return r
        elif isinstance(node, Case):
            if node.cases:
                test,testt = self._printexpr(ns, node.test, type=True)

                r = "\t" * level + "case " + test + " is \n"
                css = [(k, v) for k, v in node.cases.items() if isinstance(k, Constant)]
                css = sorted(css, key=lambda x: x[0].value)
                for choice, statements in css:
                    r += "\t" * (level + 1) + "when " + self._printliteral(choice, testt) + " =>\n"
                    r += self._printnode(ns, level + 2, statements)
                if "default" in node.cases:
                    r += "\t" * (level + 1) + "when others => \n"
                    r += self._printnode(ns, level + 2, node.cases["default"])
                r += "\t" * level + "end case;\n"
                return r
            else:
                return ""
        else:
            raise TypeError("Node of unrecognized type: " + str(type(node)))

    def _printuse(self, extra=[]):
        r = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.migen_helpers.all;
"""
        for u in extra:
            r += 'use '+u+';\n'
        return r

    def _printentitydecl(self, f, ios, name, ns,
                         reg_initialization):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        r = """
entity {name} is
\tport(
""".format(name=name)
        r += ';\n'.join(
            '\t' * 2 + self._printsig(ns, sig, 'inout' if sig in inouts else 'buffer' if sig in targets else 'in', initialise=True)
            for sig in sorted(ios, key=lambda x: x.duid)
        )
        r += """\n);
end {name};
    """.format(name=name)
        return r

    def _printarchitectureheader(self, f, ios, name, ns,
                                 reg_initialization, extra=""):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        r = """
    architecture Migen of {name} is
    """.format(name=name)
        r += '\n'.join(
            ' ' * 4 + 'signal ' + self._printsig(ns, sig, '', initialise=True) + ';'
            for sig in sorted(sigs - ios, key=lambda x: x.duid)
        ) + '\n'
        r += extra
        r += "begin\n"
        return r

    def _printsync(self, f, name, ns):
        r = ""
        for k, v in sorted(f.sync.items(), key=itemgetter(0)):
            clk = ns.get_name(f.clock_domains[k].clk)
            r += name+'_'+k+": process ({clk})\nbegin\nif rising_edge({clk}) then\n".format(clk=clk)
            r += self._printnode(ns, 1, v)
            r += "end if;end process;\n\n"
        return r

    def _printcomb(self, f, ns):
        if not f.comb:
            return '\n'

        r = ""
        groups = group_by_targets(f.comb)

        def extract_assignment(target,node,type,reset, precondition=None):
            if isinstance(node,_Assign):
                if node.l is target:
                    return self._printexpr(ns,node.r,type)
                return None
            if isinstance(node,If):
                condition = self._printexpr(ns, node.cond, integer) + '/=0'
                if precondition:
                    condition = '(' + precondition + ' and ' + condition + ')'
                return (
                    extract_assignment(target,node.t,type,reset, condition)
                    +  " when " + condition + " else "
                    + (extract_assignment(target,node.f,type,reset) if node.f else reset)
                )
            if isinstance(node,(list,tuple)):
                values = list(filter(None,[extract_assignment(target,n,type,reset) for n in node]))
                if len(values) > 1:
                    raise TypeError('More than one assignment to '+str(target))
                if not values:
                    return reset
                return values[0]
            raise TypeError('Combinatorial statements may only contain _Assign or If:\n'+format_tree(node,prefix='> '))

        for n, (target,statements) in enumerate(groups):
            for t in target:
                type = self.typeof(t)
                reset = self._printexpr(ns,t.reset, type)
                r += "\t" + ns.get_name(t) + " <= " + extract_assignment(t,statements,type,reset) + ";\n"
        r += "\n"
        return r

    def _printspecials(self, overrides, specials, ns, add_data_file):
        use = []
        decl = ""
        body = ""
        for special in sorted(specials, key=lambda x: x.duid):
            pr = call_special_classmethod(overrides, special, "emit_vhdl", self, ns, add_data_file)
            if pr is None:
                raise NotImplementedError("Special " + str(special) + " failed to implement emit_vhdl")
            use.extend(pr.get('use',[]))
            decl += pr.get('decl','')
            body += pr.get('body','')
        return dict(use=use,decl=decl,body=body)

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
        from ..sim.core import TimeManager

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
