

from ..structure import (
    Constant, Signal, ClockSignal, ResetSignal,
    _Operator,
    If,
)
from .explicit_migen_types import *
from .type_annotator import ExplicitTyper
from .types import *
from .ast import (
    Boolean,
    TestIfNonzero,
    VHDLSignal,
    Component, ComponentInstance, Port,
    Entity, EntityBody,
    NodeTransformer,
)
from .syntax import Verilog2VHDL_operator_map

literal_printers = {
    integer: lambda v, l: str(v),
    unsigned: lambda v, l: '"' + bin(v)[2:].rjust(l, '0') + '"',
    signed: lambda v, l: '"' + bin(v & ~-(1 << l))[2:].rjust(l, '0') + '"',
    std_logic: lambda v, l: "'1'" if v else "'0'",
    boolean: lambda v, l: "true" if v else "false",
}

# ---------------------------------
# integer representation converters
#
# for all integers simultaneously representable in both representations
# the conversion should be one to one. For other values the result is undefined.
def conv(template):
    def convert(repr, expr, orig=None):
        return template.format(
            x=expr,
            l=repr.indextypes[0].length if isinstance(repr, VHDLArray) else None,
        )
    return convert
integer_repr_converters = {
    (integer,std_logic):conv('to_integer(to_unsigned({x},1))'),   # '1', 'H' -> 1, else 0
    (integer,signed):conv('to_integer({x})'),   # one-to-one
    (integer,unsigned):conv('to_integer({x})'), # one-to-one
    (signed,integer):conv('to_signed({x},{l})'), # ?
    (unsigned,std_logic):conv('to_unsigned({x},{l})'),   # '1','H' -> 1, else -> 0
    (unsigned,integer):conv('to_unsigned({x},{l})'),   # ?
    (std_logic,boolean):conv("to_std_ulogic({x})"), # '1','H' -> true, else -> false
    (integer,boolean):conv("to_integer(to_unsigned(to_std_ulogic({x}),1))"), # true -> '1', false -> '0'
    (std_logic,integer):conv("get_index(to_unsigned({x},1),0)"), # truncates
    (std_logic,unsigned):conv("get_index({x},0)"), # truncates
#    (boolean,std_logic):conv("({x} = '1')"),
    (boolean,integer):conv('({x} /= 0)'),
    (boolean,unsigned):conv("(to_integer({x}) /= 0)"), # 0 -> false, else -> true
}


class VHDLPrinter(NodeTransformer):
    """ Write actual VHDL from an AST prepared with :py:class:`ToVHDLConverter`.

    Note that not all AST's are directly representable in VHDL.
    In addition to that, this converter makes some assumptions on the representation types,
    which are not necessarily checked here, if they are ensured by :py:class:`ToVHDLConverter`.
    Thus, :py:class:`VHDLPrinter` should always be used in conjunction with :py:class:`ToVHDLConverter`.

    """

    def __init__(self):
        # configuration (constants)

        # global variables

        # context variables
        pass


    def combine(self, orig, *args, **kw):
        if len(self.ancestry)>1 and isinstance(self.ancestry[-2],MigenExpression):
            # direct child of a type wrapper: replace combine call by
            wrapnode = self.ancestry[-2]
            combiner = type(self).find_handler('_wrapped_node_combiners',orig,type(self).combine_unknown_wrapped_node)
            return combiner(wrapnode, *args, **kw)
        combiner = type(self).find_handler('_node_combiners',orig,type(self).combine_unknown_node)
        return combiner(orig, *args, **kw)

    def recurse_unknown_wrapped_node(self,node):
        # TODO: what to do here?
        return self.visit(node.expr)

    @visitor_for_wrapped(Constant)
    def visit_Constant(self, node):
        printer = literal_printers[node.repr]
        return printer(node.expr.value)

    @combiner_for(TypeChange)
    def combine_TypeChange(self, node, expr, *, type=None, repr=None):
        return expr

    @combiner_for(TypeChange)
    def visit_TypeChange(self, orig, expr, *, type=None, repr=None):
        ntype = orig.type if type is None else type
        nrepr = orig.repr if repr is None else repr
        otype = orig.expr.type
        orepr = orig.expr.repr

        # TypeChange conversions are supposed to keep the represented value intact, thus they
        # depend on the interpretation of the representation.
        if not isinstance(ntype, MigenInteger) or not isinstance(otype, MigenInteger):
            # so far, only conversions between MigenInteger is supported
            raise TypeError('Undefined (representation) conversion between types %s and %s' % (otype, ntype))
        conv = integer_repr_converters.get((nrepr.ultimate_base, orepr.ultimate_base))
        if conv is not None:
            return conv(nrepr, expr, orepr)
        if not isinstance(orepr,VHDLArray) or not isinstance(nrepr,VHDLArray):
            raise TypeError('Unknown conversion between integer representations %s and %s' % (orepr, nrepr))
        if not nrepr.ulimate_base == orepr.ultimate_base:
            raise TypeError('Unknown conversion between integer representations %s and %s' % (orepr, nrepr))
        # simply assume the presence of a resize function
        expr = 'resize('+expr+','+str(type.indextypes[0].length)+')'
        if not orepr.valuetype.compatible_with(type.valuetype):
            assert orepr.castable_to(type.valuetype)
            assert type.name is not None
            expr = type.name+'('+expr+')'
        return expr

    @visitor_for(TestIfNonzero, needs_original_node=True)
    def visit_TestIfNonzero(self, orig, expr):
        if not (
            (orig.expr.repr.ultimate_base == integer)
            and (orig.repr == bool)
        ):
            raise TypeError('VHDL TestIfNonzero works only on integer and returns boolean')
        return '('+expr + ' /= 0)'

    @visitor_for_wrapped(Signal,ClockSignal,ResetSignal)
    def visit_Signal(self, node):
        raise TypeError("Bare Migen signals should be replaced with VHDLSignals first: "+str(node.expr))

    @visitor_for(Port)
    def visit_Port(self, node):
        raise NotImplementedError

    def _format_signal_decl(self, sig, initialise=False):
        ret = sig.name + ': ' + sig.repr.vhdl_repr
        if initialise:
            ret += ' := ' + self.visit(sig.reset)
        return ret

    @recursor_for(Component)
    def recurse_Component(self, node):
        # TODO: generic
        return (
            "component {name}\n"
            "\tport ({ports});"
            "end component;\n"
        ).format(
            name=node.name,
            ports=','.join('\n\t\t' + self.visit(p) for p in node.ports) + ('\n\t' if node.ports else '')
        )

    @recursor_for(ComponentInstance)
    def recurse_ComponentInstance(self, node):
        # TODO: generic map
        return "{instname}: {componentname} port map ({ports});".format(
            instname=node.name,
            componentname=node.component.name,
            ports=','.join('\n\t{port} => {signal}' + self.visit() for k,v in node.portmap.items()) + ('\n' if node.ports else '')
        )

    @recursor_for(EntityBody)
    def recurse_EntityBody(self, node):
        header = "architecture {arch} of {entity}\n".format(
            arch = node.architecture,
            entity = node.entity.name,
        )
        for com in node.components.values():
            header += self.visit(com)+'\n'

        for sig in node.signals.values():
            header += '\t'+self._format_signal_decl(sig,True)+';\n'

        ret = header + 'begin\n'
        for stmt in self.statements:
            ret += self.visit(stmt)
        ret += 'end\n'
        return ret

    @visitor_for(VHDLSignal)
    def visit_VHDLSignal(self, node):
        return node.name

    @combiner_for_wrapped(_Operator)
    def combine_Operator(self, wrapnode, op, operands):
        node = wrapnode.expr
        op = Verilog2VHDL_operator_map[op]  # TODO: op=='m'
        if len(operands) == 2:
            left, right = operands
            return '('+left + ' ' + op + ' ' + right+')'
        elif len(operands) == 1:
            # unary operators
            right, = operands
            return '(' + op + right + ')'
        else:
            raise TypeError('Unknown operator "%s" with %d operands'%(node.op,len(node.operands)))

    def visit_Slice(self, node):
        expr,type = self.visit(node.value)
        if not isinstance(type,VHDLArray):
            raise TypeError('Cannot slice value of non-array type %s'%type)
        idx, = type.indextypes
        if node.stop - node.start == 1:
            # this is not a slice, but an indexing operation!
            return expr + '(' + str(node.start) + ')', type.valuetype
        return expr + '(' + str(node.stop-1) + ' downto ' + str(node.start) + ')', type.ultimate_base[node.stop-1:node.start]

    def visit_Cat(self, node):
        pieces = []
        nbits = 0
        for o in node.l:
            expr,type = self.visit(o)
            if not isinstance(type,VHDLArray):
                pieces.append(self._convert_type(std_logic,expr,type))
 #               pieces.append(expr)
                nbits += 1
            else:
                l = type.indextypes[0].length
                pieces.append(self._convert_type(unsigned[l-1:0],expr,type))
#                pieces.append(expr)
                nbits += l
        expr = "unsigned'(" + '&'.join(reversed(pieces)) + ')';
        return expr, unsigned[nbits-1:0]

    def visit_Replicate(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_Replicate')

    def visit_Assign(self, node):
        return self._cannot_visit(node)

    def visit_If(self, node):
        return self._cannot_visit(node)

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

    def visit_Case(self, node):
        return self._cannot_visit(node)

    def visit_Fragment(self, node):
        return self._cannot_visit(node)

    def visit_statements(self, node):
        return self._cannot_visit(node)

    def visit_clock_domains(self, node):
        return self._cannot_visit(node)

    def visit_ArrayProxy(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_ArrayProxy')
        return _ArrayProxy([self.visit(choice) for choice in node.choices],
            self.visit(node.key))

    def visit_unknown(self, node):
        return self._cannot_visit(node)

    def _cannot_visit(self, node):
        raise TypeError('Node of type "%s" cannot be written as a VHDL expression'%type(node).__name__)

    def _printuse(self, extra=[]):
        r = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.migen_helpers.all;
"""
        for u in extra:
            r += 'use ' + u + ';\n'
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
            '\t' * 2 + self._printsig(ns, sig, 'inout' if sig in inouts else 'buffer' if sig in targets else 'in',
                                      initialise=True)
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
            r += name + '_' + k + ": process ({clk})\nbegin\nif rising_edge({clk}) then\n".format(clk=clk)
            r += self._printnode(ns, 1, v)
            r += "end if;end process;\n\n"
        return r

    def _printcomb(self, f, ns):
        if not f.comb:
            return '\n'

        r = ""
        groups = group_by_targets(f.comb)

        def extract_assignment(target, node, type, reset, precondition=None):
            if isinstance(node, _Assign):
                if node.l is target:
                    return self._printexpr(ns, node.r, type)
                return None
            if isinstance(node, If):
                condition = self._printexpr(ns, node.cond, integer) + '/=0'
                if precondition:
                    condition = '(' + precondition + ' and ' + condition + ')'
                return (
                    extract_assignment(target, node.t, type, reset, condition)
                    + " when " + condition + " else "
                    + (extract_assignment(target, node.f, type, reset) if node.f else reset)
                )
            if isinstance(node, (list, tuple)):
                values = list(filter(None, [extract_assignment(target, n, type, reset) for n in node]))
                if len(values) > 1:
                    raise TypeError('More than one assignment to ' + str(target))
                if not values:
                    return reset
                return values[0]
            raise TypeError(
                'Combinatorial statements may only contain _Assign or If:\n' + format_tree(node, prefix='> '))

        for n, (target, statements) in enumerate(groups):
            for t in target:
                type = self.typeof(t)
                reset = self._printexpr(ns, t.reset, type)
                r += "\t" + ns.get_name(t) + " <= " + extract_assignment(t, statements, type, reset) + ";\n"
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
        use.extend(pr.get('use', []))
        decl += pr.get('decl', '')
        body += pr.get('body', '')
    return dict(use=use, decl=decl, body=body)