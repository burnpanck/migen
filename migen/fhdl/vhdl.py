from functools import partial
from operator import itemgetter
import collections

from migen.fhdl.structure import *
from migen.fhdl.structure import _Operator, _Slice, _Assign, _Fragment
from migen.fhdl.tools import *
from migen.fhdl.visit import NodeTransformer
from migen.fhdl.namer import build_namespace
from migen.fhdl.conv_output import ConvOutput


_reserved_keywords_verilog = {
    "always", "and", "assign", "automatic", "begin", "buf", "bufif0", "bufif1",
    "case", "casex", "casez", "cell", "cmos", "config", "deassign", "default",
    "defparam", "design", "disable", "edge", "else", "end", "endcase",
    "endconfig", "endfunction", "endgenerate", "endmodule", "endprimitive",
    "endspecify", "endtable", "endtask", "event", "for", "force", "forever",
    "fork", "function", "generate", "genvar", "highz0", "highz1", "if",
    "ifnone", "incdir", "include", "initial", "inout", "input",
    "instance", "integer", "join", "large", "liblist", "library", "localparam",
    "macromodule", "medium", "module", "nand", "negedge", "nmos", "nor",
    "noshowcancelled", "not", "notif0", "notif1", "or", "output", "parameter",
    "pmos", "posedge", "primitive", "pull0", "pull1" "pulldown",
    "pullup", "pulsestyle_onevent", "pulsestyle_ondetect", "remos", "real",
    "realtime", "reg", "release", "repeat", "rnmos", "rpmos", "rtran",
    "rtranif0", "rtranif1", "scalared", "showcancelled", "signed", "small",
    "specify", "specparam", "strong0", "strong1", "supply0", "supply1",
    "table", "task", "time", "tran", "tranif0", "tranif1", "tri", "tri0",
    "tri1", "triand", "trior", "trireg", "unsigned", "use", "vectored", "wait",
    "wand", "weak0", "weak1", "while", "wire", "wor","xnor", "xor"
}

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


# -------------
# VHDL types
#  These classes represent the semantics of a type, not identity. Thus, the name of a type is not part of the class,
# nor does identity of the objects mater.
# -------------

class VHDLType:
    @property
    def name(self):
        return type(self).__name__

    def matchable_to(self,other):
        if not isinstance(other,VHDLType):
            return NotImplemented
        return type(other) == type(self)

class VHDLAccess(VHDLType):
    pass
class VHDLFile(VHDLType):
    pass

class VHDLScalar(VHDLType):
    @property
    def low(self):
        return min(self.left,self.right)

    @property
    def high(self):
        return max(self.left, self.right)

#    @abc.abstractproperty
    def length(self):
        pass

    @property
    def ascending(self):
        return self.left<self.right

class VHDLInteger(VHDLScalar):
    @property
    def length(self):
        return abs(self.right - self.left)

    def constrained(self,left,right):
        if hasattr(self,'left') or hasattr(self,'right'):
            if self.ascending:
                assert self.left<=left<right<=self.right
            else:
                assert self.left>=left>right>=self.right
        ret = type(self)()
        ret.left = left
        ret.right = right
        return ret


class VHDLReal(VHDLScalar):
    pass

class VHDLEnumerated(VHDLScalar):
    @property
    def length(self):
        return len(self.values)

class VHDLComposite(VHDLType):
    pass

class VHDLArray(VHDLComposite):
    def constrained(self,*indextypes):
        assert not hasattr(self,'indextypes')
        ret = type(self)()
        if not hasattr(ret,'valuetype'):
            ret.valuetype = self.valuetype
        ret.indextypes = tuple(indextypes)

    def matchable_to(self, other):
        if not self.valuetype.matchable_to(other.valuetype):
            return False
        if len(self.indextypes) != len(other.indextypes):
            return False
        return all(s.matchable_to(o) for s,o in zip(self.indextypes, other.indextypes))

class VHDLRecord(VHDLComposite):
    pass


# - Standard types

class bit(VHDLScalar):
    values = ('0','1')

class bit_vector(VHDLArray):
    valuetype = bit()

class boolean(VHDLEnumerated):
    values = (False,True)

class character(VHDLEnumerated):
    values = tuple(chr(k) for k in range(256)) # TODO: verify true set

class string(VHDLArray):
    type = character()

class integer(VHDLInteger):
    left,right = (-2**31+1,2**31-1)

class natural(VHDLInteger):
    left,right = (0,2**31-1)

class positive(VHDLInteger):
    left,right = (1,2**31-1)

# - std_logic_1164

class std_logic(VHDLEnumerated):
    values = tuple(c for c in 'UX01ZWLH-')

class std_logic_vector(VHDLArray):
    valuetype = std_logic()
    def __init__(self,*indextypes):
        assert all(isinstance(t,VHDLScalar) for t in indextypes)
        self.indextypes = indextypes
    @classmethod
    def make(cls,left,right):
        return cls(VHDLInteger().constrained(left,right))

class signed(std_logic_vector):
    pass

class unsigned(std_logic_vector):
    pass

# -------------------
# VHDL-specific nodes
# -------------------

class VHDLTyped:
    def __init__(self,name,type):
        assert isinstance(type,VHDLType)
        self.name = name
        self.type = type

class VHDLSignal(VHDLTyped):
    pass

class Port(VHDLTyped):
    def __init__(self,name,dir,type):
        assert dir in ['in', 'out', 'inout', 'buffer']
        super(Port,self).__init__(name,type)
        self.dir = dir

class Entity:
    def __init__(self,name,ports=[]):
        assert all(isinstance(p,Port) for p in ports)
        self.name = name
        self.ports = ports

class Component:
    def __init__(self,name,entity,**kw):
        assert not kw
        assert isinstance(entity,Entity)
        self.name = name
        self.entity = entity

class ComponentInstance:
    """An instantiation of a component.

    This is a concurrent satement.
    """
    def __init__(self,name,component,portmap={},*kw):
        assert not kw
        assert isinstance(component,Component)
        self.name = name
        self.component = component
        self.portmap = portmap


class Architecture:
    def __init__(self,entity,**kw):
        name = kw.pop('name','Migen')
        signals = kw.pop('signals',[])
        assert not kw
        assert isinstance(entity,Entity)
        assert all(isinstance(s,VHDLSignal) for s in signals)
        self.name = name
        self.entity = entity
        self.signals = signals


# ---------------------------

(_AT_BLOCKING, _AT_NONBLOCKING, _AT_SIGNAL) = range(3)


def _list_comb_wires(f):
    r = set()
    groups = group_by_targets(f.comb)
    for g in groups:
        if len(g[1]) == 1 and isinstance(g[1][0], _Assign):
            r |= g[0]
    return r

def _printcomb_verilog(f, ns,
                       display_run,
                       dummy_signal,
                       blocking_assign):
    r = ""
    if f.comb:
        if dummy_signal:
            # Generate a dummy event to get the simulator
            # to run the combinatorial process once at the beginning.
            syn_off = "// synthesis translate_off\n"
            syn_on = "// synthesis translate_on\n"
            dummy_s = Signal(name_override="dummy_s")
            r += syn_off
            r += "reg " + _printsig_verilog(ns, dummy_s) + ";\n"
            r += "initial " + ns.get_name(dummy_s) + " <= 1'd0;\n"
            r += syn_on

        groups = group_by_targets(f.comb)

        for n, g in enumerate(groups):
            if len(g[1]) == 1 and isinstance(g[1][0], _Assign):
                r += "assign " + _printnode_verilog(ns, _AT_BLOCKING, 0, g[1][0])
            else:
                if dummy_signal:
                    dummy_d = Signal(name_override="dummy_d")
                    r += "\n" + syn_off
                    r += "reg " + _printsig_verilog(ns, dummy_d) + ";\n"
                    r += syn_on

                r += "always @(*) begin\n"
                if display_run:
                    r += "\t$display(\"Running comb block #" + str(n) + "\");\n"
                if blocking_assign:
                    for t in g[0]:
                        r += "\t" + ns.get_name(t) + " = " + _printexpr_verilog(ns, t.reset)[0] + ";\n"
                    r += _printnode_verilog(ns, _AT_BLOCKING, 1, g[1])
                else:
                    for t in g[0]:
                        r += "\t" + ns.get_name(t) + " <= " + _printexpr_verilog(ns, t.reset)[0] + ";\n"
                    r += _printnode_verilog(ns, _AT_NONBLOCKING, 1, g[1])
                if dummy_signal:
                    r += syn_off
                    r += "\t" + ns.get_name(dummy_d) + " <= " + ns.get_name(dummy_s) + ";\n"
                    r += syn_on
                r += "end\n"
    r += "\n"
    return r


class TypeVisitor(NodeTransformer):
    def visit_Constant(self, node):
        return integer()

    def visit_Signal(self, node):
        if node.nbits==1:
            return std_logic()
        if node.signed:
            return signed.make(node.nbits-1,0)
        else:
            return unsigned.make(node.nbits-1,0)

    def visit_ClockSignal(self, node):
        return std_logic()

    def visit_ResetSignal(self, node):
        return std_logic()

    def visit_Operator(self, node):
        if node.op in {'&','|','^'}:
            # logical operators; in VHDL and,or,nand,nor,xor,xnor
            left,right = [self.visit(o) for o in node.operands]
            assert left.matchable_to(right)
            return left
        elif node.op in {'<','<=','==','!=','>','>='}:
            # relational operators; in VHDL <, <=, =, /=, >, >=
            left,right = [self.visit(o) for o in node.operands]
            assert left.matchable_to(right)
            return boolean()
        elif node.op in {'<<<','>>>'}:
            # shift operators; in VHDL sll,srl,sla,sra,rol,ror
            left,right = [self.visit(o) for o in node.operands]
            return left
        elif node.op in {'+','-'} and len(node.operands)==2:
            # addition operators; in VHDL +,-,&
            left,right = [self.visit(o) for o in node.operands]
            assert left.matchable_to(right)
            return left
        elif node.op in {'-'} and len(node.operands)==1:
            # unary operators; in VHDL +,-
            left, = [self.visit(o) for o in node.operands]
            return left
        elif node.op in {'*'}:
            # multiplying operators; in VHDL *,/,mod,rem
            left,right = [self.visit(o) for o in node.operands]
            assert left.matchable_to(right)    # except when physical types are involved
            return left
        elif node.op in {'~'}:
            # misc operators; in VHDL **,abs,not
            left, = [self.visit(o) for o in node.operands]
            return left
        else:
            raise TypeError('Unknown operator "%s" with %d operands'*(node.op,len(node.operands)))

    def visit_Slice(self, node):
        return _Slice(self.visit(node.value), node.start, node.stop)

    def visit_Cat(self, node):
        return Cat(*[self.visit(e) for e in node.l])

    def visit_Replicate(self, node):
        return Replicate(self.visit(node.v), node.n)

    def visit_Assign(self, node):
        return self._cannot_visit(node)

    def visit_If(self, node):
        return self._cannot_visit(node)

    def visit_Case(self, node):
        return self._cannot_visit(node)

    def visit_Fragment(self, node):
        return self._cannot_visit(node)

    def visit_statements(self, node):
        return self._cannot_visit(node)

    def visit_clock_domains(self, node):
        return self._cannot_visit(node)

    def visit_ArrayProxy(self, node):
        raise NotImplementedError
        return _ArrayProxy([self.visit(choice) for choice in node.choices],
            self.visit(node.key))

    def visit_unknown(self, node):
        return self._cannot_visit(node)

    def _cannot_visit(self, node):
        raise TypeError('Node of type "%s" has no corresponding VHDL type'%type(node).__name__)

class Converter:
    def typeof(self,obj):
        """ Calculate the VHDL type of a given expression/signal/variable. """
        return TypeVisitor().visit(obj)

    def _printsig(self, ns, s, dir):
        n = ns.get_name(s) + ': ' + dir
        if len(s) > 1:
            if s.signed:
                n += " signed"
            else:
                n += " unsigned"
            n += "(" + str(len(s) - 1) + " downto 0)";
        else:
            n += " std_logic"
        return n

    def _printconstant(self, node):
        if node.nbits == 1:
            return "'" + str(node.value) + "'", False
        v = node.value
        if node.signed:
            v += 1 << node.nbits
        v = bin(v)[2:]
        v = v.rjust(node.nbits - len(v), '0')
        return '"' + v + '"', bool(node.signed)

    def _printexpr(self, ns, node, type=None):
        if type is not None:
            if not self.typeof(node).matchable_to(type):
                val,nt = self._printexpr(ns,node,type=None)
                return (type.name+'('+val+')'),nt
        if isinstance(node, Constant):
            return self._printconstant(node)
        elif isinstance(node, Signal):
            return ns.get_name(node), node.signed
        elif isinstance(node, _Operator):
            arity = len(node.operands)
            r1, s1 = self._printexpr(ns, node.operands[0])
            if arity == 1:
                if node.op == "-":
                    if s1:
                        r = node.op + r1
                    else:
                        r = "-$signed({1'd0, " + r1 + "})"
                    s = True
                else:
                    r = node.op + r1
                    s = s1
            elif arity == 2:
                r2, s2 = self._printexpr(ns, node.operands[1])
                if node.op not in ["<<<", ">>>"]:
                    if s2 and not s1:
                        r1 = "$signed({1'd0, " + r1 + "})"
                    if s1 and not s2:
                        r2 = "$signed({1'd0, " + r2 + "})"
                r = r1 + " " + node.op + " " + r2
                s = s1 or s2
            elif arity == 3:
                assert node.op == "m"
                r2, s2 = self._printexpr(ns, node.operands[1])
                r3, s3 = self._printexpr(ns, node.operands[2])
                if s2 and not s3:
                    r3 = "$signed({1'd0, " + r3 + "})"
                if s3 and not s2:
                    r2 = "$signed({1'd0, " + r2 + "})"
                r = r1 + " ? " + r2 + " : " + r3
                s = s2 or s3
            else:
                raise TypeError
            return "(" + r + ")", s
        elif isinstance(node, _Slice):
            # Verilog does not like us slicing non-array signals...
            if isinstance(node.value, Signal) \
                    and len(node.value) == 1 \
                    and node.start == 0 and node.stop == 1:
                return self._printexpr(ns, node.value)

            if node.start + 1 == node.stop:
                sr = "[" + str(node.start) + "]"
            else:
                sr = "[" + str(node.stop - 1) + ":" + str(node.start) + "]"
            r, s = self._printexpr(ns, node.value)
            return r + sr, s
        elif isinstance(node, Cat):
            l = [self._printexpr(ns, v)[0] for v in reversed(node.l)]
            return "{" + ", ".join(l) + "}", False
        elif isinstance(node, Replicate):
            return "{" + str(node.n) + "{" + self._printexpr(ns, node.v)[0] + "}}", False
        else:
            raise TypeError("Expression of unrecognized type: '{}'".format(type(node).__name__))

    def _printnode(self, ns, at, level, node):
        if isinstance(node, _Assign):
            if at == _AT_BLOCKING:
                assignment = " := "
            elif at == _AT_NONBLOCKING:
                assignment = " <= "
            elif is_variable(node.l):
                assignment = " := "
            else:
                assignment = " <= "
            left,right = [self.typeof(o) for o in [node.l,node.r]]
            return "\t" * level + self._printexpr(ns, node.l)[0] + assignment + self._printexpr(ns, node.r, type=left)[0] + ";\n"
        elif isinstance(node, collections.Iterable):
            return "".join(list(map(partial(self._printnode, ns, at, level), node)))
        elif isinstance(node, If):
            r = "\t" * level + "if (" + self._printexpr(ns, node.cond)[0] + ") = '1' then\n"
            r += self._printnode(ns, at, level + 1, node.t)
            if node.f:
                r += "\t" * level + "else\n"
                r += self._printnode(ns, at, level + 1, node.f)
            r += "\t" * level + "end if;\n"
            return r
        elif isinstance(node, Case):
            if node.cases:
                r = "\t" * level + "case (" + self._printexpr(ns, node.test)[0] + ") is \n"
                css = [(k, v) for k, v in node.cases.items() if isinstance(k, Constant)]
                css = sorted(css, key=lambda x: x[0].value)
                for choice, statements in css:
                    r += "\t" * (level + 1) + "when (" + self._printexpr(ns, choice)[0] + ") =>\n"
                    r += self._printnode(ns, at, level + 2, statements)
                if "default" in node.cases:
                    r += "\t" * (level + 1) + "when others => \n"
                    r += self._printnode(ns, at, level + 2, node.cases["default"])
                r += "\t" * level + "end case;\n"
                return r
            else:
                return ""
        else:
            raise TypeError("Node of unrecognized type: " + str(type(node)))

    def _printuse(self):
        r = """
    library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
        """
        return r

    def _printentitydecl(self, f, ios, name, ns,
                         reg_initialization):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        wires = _list_comb_wires(f) | special_outs

        r = """
    entity {name} is
        port(
    """.format(name=name)
        r += ';\n'.join(
            ' ' * 8 + self._printsig(ns, sig, 'inout' if sig in inouts else 'out' if sig in targets else 'in')
            for sig in sorted(ios, key=lambda x: x.duid)
        )
        r += """\n);
    end {name};
    """.format(name=name)
        return r

    def _printarchitectureheader(self, f, ios, name, ns,
                                 reg_initialization):
        sigs = list_signals(f) | list_special_ios(f, True, True, True)
        special_outs = list_special_ios(f, False, True, True)
        inouts = list_special_ios(f, False, False, True)
        targets = list_targets(f) | special_outs
        wires = _list_comb_wires(f) | special_outs
        r = """
    architecture Migen of {name} is
    """.format(name=name)
        r += '\n'.join(
            ' ' * 4 + 'signal ' + self._printsig(ns, sig, '') + ';'
            for sig in sorted(sigs - ios, key=lambda x: x.duid)
        )
        r += "\nbegin\n"
        return r

    def _printsync(self, f, ns):
        r = ""
        for k, v in sorted(f.sync.items(), key=itemgetter(0)):
            r += "WENEEDANAME: process (" + ns.get_name(f.clock_domains[k].clk) + ")\nbegin\n"
            r += self._printnode(ns, _AT_SIGNAL, 1, v)
            r += "end process;\n\n"
        return r

    def convert(self, f, ios=None, name="top",
      special_overrides=dict(),
      create_clock_domains=True,
      display_run=False, asic_syntax=False):
        r = ConvOutput()
        if not isinstance(f, _Fragment):
            f = f.get_fragment()
        if ios is None:
            ios = set()

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

        f = lower_complex_slices(f)
        insert_resets(f)
        f = lower_basics(f)
        fs, lowered_specials = lower_specials(special_overrides, f.specials)
        f += lower_basics(fs)

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

        src = "-- Machine-generated using Migen\n"
        src += self._printuse()
        src += self._printentitydecl(f, ios, name, ns, reg_initialization=not asic_syntax)
        src += self._printarchitectureheader(f, ios, name, ns, reg_initialization=not asic_syntax)
        src += self._printsync(f, ns)
        src += "end Migen;\n"
        if False:
            src += _printheader_verilog(f, ios, name, ns,
                                        reg_initialization=not asic_syntax)
            src += _printcomb_verilog(f, ns,
                                      display_run=display_run,
                                      dummy_signal=not asic_syntax,
                                      blocking_assign=asic_syntax)
            src += _printsync_verilog(f, ns)
            src += _printspecials_verilog(special_overrides, f.specials - lowered_specials, ns, r.add_data_file)
            src += "endmodule\n"
        r.set_main_source(src)

        return r

def convert(f, ios=None, name="top", special_overrides={}, create_clock_domains=True, display_run=False, asic_syntax=False):
    return Converter().convert(f,ios,name,special_overrides,create_clock_domains,display_run,asic_syntax)